import gradio as gr
from PIL import Image
import torch
from core.model import Generator
from torchvision import transforms

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 256
style_dim = 64
num_domains = 3  # Cambia según tu dataset (por ejemplo, CelebA-HQ usa 3: hombre, mujer, viejo)

# Carga del generador
G = Generator(img_size, style_dim, num_domains).to(device)
G.load_state_dict(torch.load("generator.pth", map_location=device))
G.eval()

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def generate(img, domain):
    x = transform(img).unsqueeze(0).to(device)
    style_code = torch.randn(1, style_dim).to(device)  # estilo aleatorio
    with torch.no_grad():
        out = G(x, style_code, torch.tensor([domain]).to(device))
    out = out.squeeze(0).cpu().detach()
    out = (out * 0.5 + 0.5).clamp(0, 1)  # desnormalizar
    out = transforms.ToPILImage()(out)
    return out

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Image(label="Imagen de entrada", type="pil"),
        gr.Radio(choices=[0, 1, 2], label="Dominio de estilo (0=hombre, 1=mujer, 2=viejo)")
    ],
    outputs=gr.Image(label="Imagen generada"),
    title="StarGAN v2 - Rostros de celebridades",
    description="Selecciona una imagen y el dominio de estilo para generar un nuevo rostro con StarGAN v2."
)

demo.launch()
