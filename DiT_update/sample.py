# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Enhanced DiT sampling with medical anomaly detection capabilities.
"""
import os
import torch
import argparse
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import torch.nn.functional as F

# Configure PyTorch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class MedicalClassifierWrapper(torch.nn.Module):
    """
    Example classifier wrapper for medical anomaly detection.
    Replace with your actual classifier implementation.
    """

    def __init__(self, num_classes=2):
        super().__init__()
        # Example architecture - replace with your actual classifier
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1)
        )
        self.fc = torch.nn.Linear(128, num_classes)

    def forward(self, x, t=None):
        # t is ignored in this simple example but can be used for time conditioning
        features = self.conv(x)
        return self.fc(features.squeeze())


def load_classifier(ckpt_path, device):
    """Load the medical classifier model"""
    classifier = MedicalClassifierWrapper().to(device)
    if ckpt_path:
        state_dict = torch.load(ckpt_path, map_location=device)
        classifier.load_state_dict(state_dict)
    classifier.eval()
    return classifier


def detect_anomaly(diffusion, model, classifier, vae, image, noise_level=500, classifier_scale=100.0):
    """
    Complete anomaly detection pipeline:
    1. VAE encode the image to latent space
    2. DDIM encode to specified noise level
    3. Generate healthy reconstruction with classifier guidance
    4. Compute anomaly map
    """
    with torch.no_grad():
        # 1. Encode to latent space
        latent = vae.encode(image).latent_dist.sample() * 0.18215

        # 2. DDIM encode to noise level
        encoded = diffusion.ddim_encode_loop(model, latent, noise_level)

        # 3. Generate healthy reconstruction - NOW WITH PROPER PARENTHESES
        healthy_result = diffusion.ddim_sample_with_classifier_guidance(
            model,
            encoded,
            torch.tensor([noise_level - 1] * len(image), device=image.device),
            classifier=classifier,
            classifier_scale=classifier_scale
        )
        healthy_latent = healthy_result['sample']

        # 4. Decode both original and healthy images
        healthy_img = vae.decode(healthy_latent / 0.18215).sample

        # 5. Compute anomaly map (perceptual difference)
        anomaly_map = F.l1_loss(image, healthy_img, reduction='none').mean(dim=1, keepdim=True)

        # Normalize for visualization
        anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)

        return {
            'healthy_image': healthy_img,
            'anomaly_map': anomaly_map,
            'encoded': encoded
        }


def main(args):
    # Setup PyTorch
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Validate parameters
    if args.run_anomaly_detection and args.noise_level > args.num_sampling_steps:
        raise ValueError(f"Noise level {args.noise_level} exceeds num_sampling_steps {args.num_sampling_steps}")

    # Load models
    print("Loading base models...")
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)

    # Load weights
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Load classifier if doing anomaly detection
    classifier = None
    if args.run_anomaly_detection:
        print("Loading medical classifier...")
        classifier = load_classifier(args.classifier_ckpt, device)

    # Sampling parameters
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]  # Example labels
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Generate samples
    print("Generating samples...")
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save original samples
    os.makedirs("outputs", exist_ok=True)
    save_image(
        samples,
        "outputs/generated_samples.png",
        nrow=4,
        normalize=True,
        value_range=(-1, 1)
    )

    # Run anomaly detection if requested
    if args.run_anomaly_detection:
        print("Running anomaly detection...")
        samples_for_ad = samples.clamp(-1, 1)  # Ensure proper range

        results = detect_anomaly(
            diffusion,
            model,
            classifier,
            vae,
            samples_for_ad,
            noise_level=args.noise_level,
            classifier_scale=args.classifier_scale
        )

        # Save results
        save_image(
            results['healthy_image'],
            "outputs/healthy_reconstructions.png",
            nrow=4,
            normalize=True,
            value_range=(-1, 1)
        )

        # Save anomaly map as heatmap (convert to RGB)
        anomaly_rgb = results['anomaly_map'].repeat(1, 3, 1, 1)
        save_image(
            anomaly_rgb,
            "outputs/anomaly_maps.png",
            nrow=4,
            normalize=True
        )

        print("Anomaly detection results saved to outputs/ directory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Sampling parameters
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to DiT checkpoint (default: auto-download)")

    # Anomaly detection parameters
    parser.add_argument("--run-anomaly-detection", action="store_true",
                        help="Enable medical anomaly detection pipeline")
    parser.add_argument("--noise-level", type=int, default=500,
                        help="DDIM encoding steps for anomaly detection")
    parser.add_argument("--classifier-scale", type=float, default=100.0,
                        help="Classifier guidance scale")
    parser.add_argument("--classifier-ckpt", type=str, default=None,
                        help="Path to medical classifier checkpoint")

    args = parser.parse_args()
    main(args)