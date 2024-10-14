import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from vqgan import VQGAN

class TestVqgan:
    def __init__(self, args):
        self.vqgan = VQGAN(args).to(device=args.device)
        self.vqgan.eval()  # Set the model to evaluation mode
        self.load_weights(args.checkpoint_path)

    def load_weights(self, checkpoint_path):
        if os.path.isfile(checkpoint_path):
            print(f"Loading weights from {checkpoint_path}")
            self.vqgan.load_state_dict(torch.load(checkpoint_path))
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    def load_images(self, folder_path):
        images = []
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Adjust size as needed
            transforms.ToTensor(),
        ])
        for img_name in os.listdir(folder_path):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, img_name)
                image = Image.open(img_path).convert('RGB')
                image = transform(image)
                images.append(image)
        return images

    def generate_decoded_images(self, images):
        with torch.no_grad():
            images_tensor = torch.stack(images).to(device=args.device)
            decoded_images, _, _ = self.vqgan(images_tensor)
            return decoded_images

    def save_decoded_images(self, decoded_images, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        for i, decoded_img in enumerate(decoded_images):
            # Convert tensor to PIL image
            img = transforms.ToPILImage()(decoded_img.cpu())
            img.save(os.path.join(output_folder, f"decoded_{i}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQGAN Testing Parameters")
    parser.add_argument("--checkpoint_path", type=str, default = "/mnt/VQGAN_from_scratch/checkpoints/vqgan_epoch_100.pt", help="Path to the trained VQGAN checkpoint")
    parser.add_argument("--image_folder", type=str, default = "/mnt/VQGAN_from_scratch/input", help="Path to the folder containing input images")
    parser.add_argument("--output_folder", type=str, default = "/mnt/VQGAN_from_scratch/output" , help="Path to save decoded images")
    parser.add_argument("--latent_dim", type = int, default = 256, help = "latent dimension of vqgan")
    parser.add_argument("--image_size", type = int, default = 256, help = "image size for dataset")
    parser.add_argument("--num_codebook_vectors", type = int, default = 1024, 
                        help = "number of vqgan codebook vectors")
    parser.add_argument("--beta", type = float, default = 0.25, help = "commitment loss scalar")
    parser.add_argument("--image_channels", type = int, default = 3, help = "number of channels of images")
    parser.add_argument("--dataset_path", type = str, default = "./dataset", 
                        help = "path for dataset to save data")
    parser.add_argument("--vqgan_model_path", type = str, default = "./pretrained_models/vqgan_epoch_308_flower.pt", 
                        help = "path for pretrained vqgan models")
    parser.add_argument("--discriminator_model_path", type = str, default = "./pretrained_models/discriminator_epoch_308_flower.pt", help = "path for pretrained discriminator models")
    parser.add_argument("--device", type = str, default = "cuda", help = "which device to train the data, cuda or cpu")
    parser.add_argument("--batch_size", type = int, default = 16, help = "batch size for training model")
    parser.add_argument("--num_epochs", type = int, default = 1000, help = "number of epoches to train")
    parser.add_argument("--learning_rate", type = float, default = 2.25e-5, help = "learning rate")
    parser.add_argument("--beta1", type = float, default = 0.5, help = "adam beta parmeters")
    parser.add_argument("--beta2", type = float, default = 0.9, help = "adam beta parmeters")
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=0.2, help='')
    parser.add_argument('--rec_loss_factor', type=float, default=1.0, help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual_loss_factor', type=float, default=1.0,
                        help='Weighting factor for perceptual loss.')

    args = parser.parse_args()

    tester = TestVqgan(args)
    images = tester.load_images(args.image_folder)
    decoded_images = tester.generate_decoded_images(images)
    tester.save_decoded_images(decoded_images, args.output_folder)
    print("Decoding completed and images saved.")
