import gradio as gr
from transformers import AutoProcessor, AutoModelForCausalLM
# import peft
import spaces

import requests
import copy

from PIL import Image, ImageDraw, ImageFont 
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import random
import numpy as np

import subprocess
subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

models = {
    'microsoft/Florence-2-large-ft': AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large-ft', trust_remote_code=True).to("cuda").eval(),
    'dwb2023/florence2-large-bccd-base-ft': AutoModelForCausalLM.from_pretrained('dwb2023/florence2-large-bccd-base-ft', trust_remote_code=True).to("cuda").eval(),
}

processors = {
    'microsoft/Florence-2-large-ft': AutoProcessor.from_pretrained('microsoft/Florence-2-large-ft', trust_remote_code=True),
    'dwb2023/florence2-large-bccd-base-ft': AutoProcessor.from_pretrained('microsoft/Florence-2-large-ft', trust_remote_code=True),  
}

colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return Image.open(buf)

# spaces.GPU
def run_example(task_prompt, image, text_input=None, model_id='microsoft/Florence-2-large'):
    model = models[model_id]
    processor = processors[model_id]
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer

def plot_bbox(image, data):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
    ax.axis('off')
    return fig

def draw_polygons(image, prediction, fill_mask=False):

    draw = ImageDraw.Draw(image)
    scale = 1
    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        color = random.choice(colormap)
        fill_color = random.choice(colormap) if fill_mask else None
        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
                continue
            _polygon = (_polygon * scale).reshape(-1).tolist()
            if fill_mask:
                draw.polygon(_polygon, outline=color, fill=fill_color)
            else:
                draw.polygon(_polygon, outline=color)
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)
    return image

def convert_to_od_format(data):
    bboxes = data.get('bboxes', [])
    labels = data.get('bboxes_labels', [])
    od_results = {
        'bboxes': bboxes,
        'labels': labels
    }
    return od_results

def draw_ocr_bboxes(image, prediction):
    scale = 1
    draw = ImageDraw.Draw(image)
    bboxes, labels = prediction['quad_boxes'], prediction['labels']
    for box, label in zip(bboxes, labels):
        color = random.choice(colormap)
        new_box = (np.array(box) * scale).tolist()
        draw.polygon(new_box, width=3, outline=color)
        draw.text((new_box[0]+8, new_box[1]+2),
                  "{}".format(label),
                  align="right",
                  fill=color)
    return image

def process_image(image, task_prompt, text_input=None, model_id='dwb2023/florence2-large-bccd-base-ft'):
    image = Image.fromarray(image)  # Convert NumPy array to PIL Image
    if task_prompt == 'Object Detection':
        task_prompt = '<OD>'
        results = run_example(task_prompt, image, model_id=model_id)
        fig = plot_bbox(image, results['<OD>'])
        return results, fig_to_pil(fig)
    else:
        return "", None  # Return empty string and None for unknown task prompts

single_task_list =[
    'Object Detection'
]

with gr.Blocks(theme="sudeepshouche/minimalist") as demo:
    gr.Markdown("## 🧬OmniScience - fine tuned VLM models for use in function calling 🔧")
    gr.Markdown("- 🔬Florence-2 Model Proof of Concept, focusing on Object Detection <OD> tasks.")
    gr.Markdown("- Fine-tuned for 🩸Blood Cell Detection using the [Roboflow BCCD dataset](https://universe.roboflow.com/roboflow-100/bccd-ouzjz/dataset/2), this model can detect blood cells and types in images.")
    gr.Markdown("")
    gr.Markdown("BCCD Datasets on Hugging Face:")
    gr.Markdown("- [🌺Florence 2](https://huggingface.co/datasets/dwb2023/roboflow100-bccd-florence2/viewer/default/test?q=BloodImage_00038_jpg.rf.1b0ce1635e11b3b49302de527c86bb02.jpg), [💎PaliGemma](https://huggingface.co/datasets/dwb2023/roboflow-bccd-paligemma/viewer/default/test?q=BloodImage_00038_jpg.rf.1b0ce1635e11b3b49302de527c86bb02.jpg)")


    with gr.Tab(label="Florence-2 Object Detection"):
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(label="Input Picture")
                model_selector = gr.Dropdown(choices=list(models.keys()), label="Model", value='microsoft/Florence-2-large-ft')
                task_prompt = gr.Dropdown(choices=single_task_list, label="Task Prompt", value="Object Detection")
                text_input = gr.Textbox(label="Text Input", placeholder="Not used for Florence-2 Object Detection")
                submit_btn = gr.Button(value="Submit")
            with gr.Column():
                output_text = gr.Textbox(label="Output Text")
                output_img = gr.Image(label="Output Image")

        gr.Examples(
            examples=[
                ["examples/bccd-test/BloodImage_00038_jpg.rf.1b0ce1635e11b3b49302de527c86bb02.jpg", 'Object Detection'],
                ["examples/bccd-test/BloodImage_00044_jpg.rf.1c44102fcdf64fd178f1f16bb988d5cf.jpg", 'Object Detection'],
                ["examples/bccd-test/BloodImage_00062_jpg.rf.fbed5373cd2e0e732092ed5c7b28aa19.jpg", 'Object Detection'],
                ["examples/bccd-test/BloodImage_00090_jpg.rf.7e3d419774b20ef93d4ec6c4be8f64df.jpg", 'Object Detection'],
                ["examples/bccd-test/BloodImage_00099_jpg.rf.0a65e56401cdd71253e7bc04917c3558.jpg", 'Object Detection'],
                ["examples/bccd-test/BloodImage_00112_jpg.rf.6b8d185de08e65c6d765c824bb76ec68.jpg", 'Object Detection'],
                ["examples/bccd-test/BloodImage_00113_jpg.rf.ab69dfaa52c1b3249cf44fa66afbb619.jpg", 'Object Detection'],
                ["examples/bccd-test/BloodImage_00120_jpg.rf.4a2f84ca3564ef453b12ceb9c852e32e.jpg", 'Object Detection'],
            ],
            
            inputs=[input_img, task_prompt],
            outputs=[output_text, output_img],
            fn=process_image,
            cache_examples=True,
            label='Try examples'
        )

        submit_btn.click(process_image, [input_img, task_prompt, model_selector], [output_text, output_img])

    gr.Markdown("## 🚀Other Cool Stuff:")
    gr.Markdown("- [Florence 2 Whitepaper](https://arxiv.org/pdf/2311.06242) - how I found out about the Roboflow 100 and the BCCD dataset.")
    gr.Markdown("- [Roboflow YouTube Video on Florence 2 fine-tuning](https://youtu.be/i3KjYgxNH6w?si=x1ZMg9hsNe25Y19-&t=1296) - bookmarked an 🧠insightful trade-off analysis of various VLMs.")
    gr.Markdown("- [Landing AI - Vision Agent](https://va.landing.ai/) - 🌟just pure WOW.  bringing agentic planning into solutions architecture.")
    gr.Markdown("- [OmniScience fork of Landing AI repo](https://huggingface.co/spaces/dwb2023/omniscience) - I had a lot of fun with this one... some great 🔍reverse engineering enabled by W&B's Weave📊.")
    gr.Markdown("- [Scooby Snacks🐕  - microservice based function calling with style](https://huggingface.co/spaces/dwb2023/blackbird-app) - Leveraging 🤖Claude Sonnet 3.5 to orchestrate Microservice-Based Function Calling.")

demo.launch(debug=True)
