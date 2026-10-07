import torch # Ejecutar operaaciones matematicas del modelo de IA
from diffusers import AutoPipelineForText2Image # Diffusers es una libreria especializada en modelos generativos

print("Cargando el modelo ...")

modelo = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32 # Float16 o float32
    #torch.floatXX, indica la presición numerica con que se cargan los pesos (decimales, la presición para generar la imagen)
)

modelo = modelo.to("cpu")

prompt = input("Escribe el prompt de la imagen que quieres crear: ")

# negative_prompt = "blurry, low quality, low resolution, pixelated, jpeg artifacts, noise, grainy, deformed, distorted, disfigured, bad anatomy, extra limbs, extra fingers, missing fingers, fused fingers, mutated hands, poorly drawn face, asymmetric eyes, cropped, out of frame, watermark, text, signature, logo, oversaturated, overexposed, duplicate"

print("Generando imagen ...")

imagen = modelo(
    prompt=prompt, # Instrucciones o detalle para crear la imagen
    # negative_prompt = negative_prompt, 
    num_inference_step=25, # Las veces que va a tratar de crear o refinar la imagen
    guidance_scale=7.0, # Que tan fiel debe ser al prompt (creatividad, temperatura)
    hight=1024,
    width=1024
).images[0]

imagen.save("imagen.png")

print("Imagen gurdada.")
