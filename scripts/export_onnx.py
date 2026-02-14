"""Export a trained PPO model to ONNX format for sim-to-real transfer.

Extracts the policy network from an SB3 PPO model and exports it
to ONNX format using torch.onnx.export. Supports Dict observation
spaces (image + velocity) via a policy wrapper.

Usage:
    python -m scripts.export_onnx --model logs/ppo/best_model/best_model.zip
    python -m scripts.export_onnx --model logs/ppo/best_model/best_model.zip --output model.onnx
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO


class PolicyWrapper(torch.nn.Module):
    """Wraps SB3 policy to expose Dict obs inputs for ONNX export.

    Handles the feature extraction and action prediction steps
    in a single forward pass with separate image and velocity inputs.
    """

    def __init__(self, policy):
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, image, velocity):
        """Forward pass with image and velocity tensors.

        Args:
            image: Tensor of shape (batch, 84, 84, frame_stack)
            velocity: Tensor of shape (batch, 3)

        Returns:
            action_mean: Tensor of shape (batch, 3)
        """
        obs = {"image": image, "velocity": velocity}
        features = self.features_extractor(obs)
        latent_pi, _ = self.mlp_extractor(features)
        action_mean = self.action_net(latent_pi)
        return action_mean


def export_to_onnx(
    model_path: str,
    output_path: str,
    frame_stack: int = 4,
    verbose: bool = True,
) -> bool:
    """Export SB3 PPO policy to ONNX format.

    Args:
        model_path: Path to trained .zip model file
        output_path: Path to save .onnx model
        frame_stack: Number of stacked depth frames (default 4)
        verbose: Print detailed logging (default True)

    Returns:
        True if export successful, False otherwise
    """
    model_path = str(model_path)
    output_path = str(output_path)

    if verbose:
        print(f"[export] Loading model from {model_path}")

    if not os.path.exists(model_path):
        print(f"[export] ERROR: Model not found at {model_path}")
        return False

    try:
        model = PPO.load(model_path, device="cpu")
    except Exception as e:
        print(f"[export] ERROR: Failed to load model: {e}")
        return False

    policy = model.policy
    policy.eval()

    if verbose:
        print(f"[export] Observation space: {model.observation_space}")
        print(f"[export] Action space: {model.action_space}")

    # Validate observation space structure
    try:
        assert "image" in model.observation_space.spaces
        assert "velocity" in model.observation_space.spaces
        image_shape = model.observation_space.spaces["image"].shape
        velocity_shape = model.observation_space.spaces["velocity"].shape
        assert image_shape == (84, 84, frame_stack), (
            f"Expected image shape (84, 84, {frame_stack}), "
            f"got {image_shape}"
        )
        assert velocity_shape == (3,), (
            f"Expected velocity shape (3,), got {velocity_shape}"
        )
    except AssertionError as e:
        print(f"[export] ERROR: Observation space mismatch: {e}")
        return False

    # Create dummy inputs
    dummy_image = torch.randn(1, 84, 84, frame_stack, dtype=torch.float32)
    dummy_velocity = torch.randn(1, 3, dtype=torch.float32)

    # Test inference with SB3's predict method
    try:
        with torch.no_grad():
            action_np, _ = model.predict(
                {
                    "image": dummy_image.numpy(),
                    "velocity": dummy_velocity.numpy(),
                },
                deterministic=True,
            )
        if verbose:
            print(f"[export] SB3 test inference: {action_np}")
    except Exception as e:
        print(f"[export] ERROR: SB3 inference failed: {e}")
        return False

    # Save PyTorch state dict as fallback
    torch_path = output_path.replace(".onnx", ".pt")
    try:
        torch.save(
            {
                "policy_state_dict": policy.state_dict(),
                "observation_space": str(model.observation_space),
                "action_space": str(model.action_space),
                "frame_stack": frame_stack,
            },
            torch_path,
        )
        if verbose:
            print(f"[export] Saved PyTorch state dict to {torch_path}")
    except Exception as e:
        print(f"[export] WARNING: Failed to save PyTorch fallback: {e}")

    # Export to ONNX
    try:
        wrapper = PolicyWrapper(policy)
        wrapper.eval()

        torch.onnx.export(
            wrapper,
            (dummy_image, dummy_velocity),
            output_path,
            input_names=["image", "velocity"],
            output_names=["action"],
            dynamic_axes={
                "image": {0: "batch"},
                "velocity": {0: "batch"},
                "action": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
            verbose=False,
        )
        if verbose:
            print(f"[export] Saved ONNX model to {output_path}")
    except Exception as e:
        print(f"[export] ERROR: ONNX export failed: {e}")
        print(f"[export] PyTorch fallback available at {torch_path}")
        return False

    # Validate ONNX export
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(output_path)
        onnx_result = session.run(
            None,
            {
                "image": dummy_image.numpy(),
                "velocity": dummy_velocity.numpy(),
            },
        )
        onnx_action = onnx_result[0]

        if verbose:
            print(f"[export] ONNX inference shape: {onnx_action.shape}")
            print(f"[export] ONNX inference output: {onnx_action}")

        # Compare PyTorch vs ONNX outputs (should be close)
        torch_action = action_np
        max_diff = np.abs(onnx_action - torch_action).max()
        mean_diff = np.abs(onnx_action - torch_action).mean()

        if verbose:
            print(f"[export] PyTorch vs ONNX max diff: {max_diff:.6f}")
            print(f"[export] PyTorch vs ONNX mean diff: {mean_diff:.6f}")

        if max_diff > 1e-3:
            print(
                f"[export] WARNING: Large difference between PyTorch "
                f"and ONNX outputs (max={max_diff:.6f})"
            )

    except ImportError:
        print(
            "[export] WARNING: onnxruntime not installed; skipping ONNX "
            "validation. Install with: pip install onnxruntime"
        )
    except Exception as e:
        print(f"[export] ERROR: ONNX validation failed: {e}")
        return False

    print(f"[export] Success! Model exported to {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export trained PPO model to ONNX format"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained .zip model file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .onnx path (default: derived from model path)",
    )
    parser.add_argument(
        "--frame_stack",
        type=int,
        default=4,
        help="Frame stack depth (default 4)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(args.model)[0]
        args.output = f"{base}.onnx"

    success = export_to_onnx(
        args.model,
        args.output,
        args.frame_stack,
        verbose=not args.quiet,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
