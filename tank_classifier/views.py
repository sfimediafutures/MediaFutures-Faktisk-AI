from django.shortcuts import render
from django.http import HttpResponse
import base64
import io
import json
import os
import timm
from torchvision import models
from torchvision import transforms
from PIL import Image
from django.shortcuts import render
from django.conf import settings
import torch
from .forms import ImageUploadForm, UploadFileForm
from django.core.files.storage import FileSystemStorage
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from PIL import Image
from torchvision import datasets, transforms
import numpy as np
import base64
import pandas as pd

import whisper
LANGUAGES = {
    "en": "english", "zh": "chinese", "de": "german","es": "spanish","ru": "russian","ko": "korean","fr": "french",
    "ja": "japanese","pt": "portuguese","tr": "turkish","pl": "polish","ca": "catalan","nl": "dutch","ar": "arabic",
    "sv": "swedish","it": "italian","id": "indonesian","hi": "hindi","fi": "finnish","vi": "vietnamese","iw": "hebrew",
    "uk": "ukrainian","el": "greek","ms": "malay","cs": "czech","ro": "romanian","da": "danish","hu": "hungarian",
    "ta": "tamil","no": "norwegian","th": "thai","ur": "urdu","hr": "croatian","bg": "bulgarian","lt": "lithuanian",
    "la": "latin","mi": "maori","ml": "malayalam","cy": "welsh","sk": "slovak","te": "telugu","fa": "persian","lv": "latvian",
    "bn": "bengali","sr": "serbian","az": "azerbaijani","sl": "slovenian","kn": "kannada","et": "estonian","mk": "macedonian",
    "br": "breton","eu": "basque","is": "icelandic","hy": "armenian","ne": "nepali","mn": "mongolian","bs": "bosnian","kk": "kazakh",
    "sq": "albanian","sw": "swahili","gl": "galician","mr": "marathi","pa": "punjabi","si": "sinhala","km": "khmer",
    "sn": "shona","yo": "yoruba","so": "somali","af": "afrikaans","oc": "occitan","ka": "georgian","be": "belarusian",
    "tg": "tajik","sd": "sindhi","gu": "gujarati","am": "amharic","yi": "yiddish","lo": "lao","uz": "uzbek","fo": "faroese",
    "ht": "haitian creole","ps": "pashto","tk": "turkmen","nn": "nynorsk","mt": "maltese","sa": "sanskrit","lb": "luxembourgish",
    "my": "myanmar","bo": "tibetan","tl": "tagalog","mg": "malagasy","as": "assamese","tt": "tatar","haw": "hawaiian",
    "ln": "lingala","ha": "hausa","ba": "bashkir","jw": "javanese","su": "sundanese",
}

langauge_classification_model = whisper.load_model("small").to('cpu')

# Create your views here.
# def home(request):
# 	return render(request, 'tank_classifier/home.html')

# labels_map = ["Armata", "Challenger 2", "K2 Black Panther", "Leclerc", "Leopard 2", "M1A2 Abrams", "Merkava Mk.4", 
#               "Oplot-M", "T-90", "Type 90"]

labels_map = ['Al Khalid', 'Arjun', 'Armata', 'Challenger 2', 'K2 Black Panther', 'K9 Thunder', 'Leclerc', 
    'Leopard 2', 'M109', 'M1A2 Abrams', 'Merkava Mk.4', 'Oplot-M', 'T-55', 'T-72', 'T-80', 'T-90', 
    'Type 74', 'Type 90', 'Type 99', 'VT-4']

# PyTorch-related code from: https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html
# load pretrained DenseNet and go straight to evaluation mode for inference
# load as global variable here, to avoid expensive reloads with each request

model = timm.create_model('res2net50_26w_8s', pretrained=True, num_classes=20)
weights_path = os.path.join(settings.STATICFILES_DIRS[0], "20_mbt_classifier_res2net_43_epochs.pth")
model.load_state_dict(torch.load(weights_path))
model.eval()


# load mapping of ImageNet index to human-readable label
# run "python manage.py collectstatic" first!
# json_path = os.path.join(settings.STATICFILES_DIRS[0], "imagenet_class_index.json")
# json_path = os.path.join(settings.STATIC_ROOT, "imagenet_class_index.json")
# imagenet_mapping = json.load(open(json_path))

def get_prediction(request, image):
    
    fileObj = request.FILES['image']
    fs = FileSystemStorage()
    filePathName = fs.save(image, fileObj)
    filePathName = fs.url(filePathName)
    testimage = '.' + filePathName
    img = Image.open(testimage).convert('RGB')
    tfms = transforms.Compose([transforms.Resize((384,384)), transforms.ToTensor(), 
                           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
    tensor = tfms(img).unsqueeze(0)
    top_labels = []
    probabilities = []
    probs = {}
    # tensor = transform_image(img)
    outputs = model(tensor)
    for idx in torch.topk(outputs[0], k=10).indices.tolist():
        prob = torch.softmax(outputs[0], 0)[idx].item()
        probs[idx] = (labels_map[idx], prob*100)
        top_labels.append(labels_map[idx])
        probabilities.append(round(prob*100, 2))

    tfms = transforms.Compose([transforms.Resize((384,384)), transforms.ToTensor(), ])
    tensor = tfms(img).unsqueeze(0)
    tensor = tensor.squeeze(0)
    np_arr = tensor.cpu().detach().numpy()
    np_arr = np_arr/np.amax(np_arr) # if float
    np_arr = np_arr.transpose(1,2,0)
    target_layers = [model.layer4]
    input_tensor = tensor.unsqueeze(0)

    # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!

    # Construct the CAM object once, and then re-use it on many images:
    cam = HiResCAM(model=model, target_layers=target_layers, use_cuda=False)
    targets = [ClassifierOutputTarget(1)]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=None, aug_smooth=True, eigen_smooth=True)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(np_arr, grayscale_cam, use_rgb=True)
    im = Image.fromarray(np.uint8(visualization))
    filePathName = filePathName.split(".")[0].split("/")[-1]
    image_path = os.path.join(settings.STATICFILES_DIRS[1], "")
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    image = im.save(f"{image_path}/{filePathName}.png")
    image_name = image_path+"/"+filePathName + ".png"
    image_name = image_name.replace('/','\\')
    image_name = image_name.split("/")[-1]
    # image_name = image_name.replace('\\','/')

    tanks_dataframe = pd.read_csv(os.path.join(settings.STATICFILES_DIRS[0], 'tanks_details.csv'))
    row = tanks_dataframe.loc[tanks_dataframe['Name'] == top_labels[0]]
    tank_dict = row.to_dict('index')

    image_name = None
    return top_labels[:5], probabilities[:5], tank_dict, image_name


def tank_classifier_view(request):
    image_uri = None
    predicted_label = None
    probs = None
    visualization = None
    tank_dict = None
    image_name = None
    lisst = []
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image_bytes = image.file.read()
            encoded_img = base64.b64encode(image_bytes).decode('ascii')
            image_uri = 'data:%s;base64,%s' % ('image/jpeg', encoded_img)
            # get predicted label
            try:
                predicted_label, probs, tank_dict, image_name = get_prediction(request, image)
                lisst.extend([list(a) for a in zip(predicted_label, probs)])
                # encoded_img = base64.b64encode(im.tobytes())
                # byte_image = 'data:%s;base64,%s' % ('image/png', encoded_img)
            except RuntimeError as re:
                print(re)
                # predicted_label = "Prediction Error"
    else:
        form = ImageUploadForm()

    context = {
        'form': form,
        'image_uri': image_uri,
        'predicted_label': predicted_label,
        'probabilities': probs,
        'dictionary': tank_dict,
        'image_name': image_name,
        'predictions': lisst,
    }
    return render(request, 'tank_classifier/home.html', context)


def langauge_classifier_view(request):
    lang = None
    result_translated = None
    transcription = None
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        uploaded_file_url = uploaded_file_url.replace('/', '\\')
        file_path_refined = uploaded_file_url.split('\\')[-1]
        audio_path = os.path.join(settings.MEDIA_ROOT, file_path_refined)
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio)

        # detect the spoken language
        lang = None
        _, probs = langauge_classification_model.detect_language(mel)
        lang = LANGUAGES[max(probs, key=probs.get)].capitalize()
        options = whisper.DecodingOptions(task="tranlsate", fp16 = False)
        result_translated = whisper.decode(langauge_classification_model, mel, options)
        result_translated = result_translated.text.capitalize()
        # transcription = langauge_classification_model.transcribe(audio_path)
        # transcription = transcription["text"]
        print(lang)

        # return render(request, 'tank_classifier/check_language.html', {
        #     'language': lang,
        # })
    context = {
            'language': lang,
            'text': transcription,
            'translation': result_translated,
        }
    return render(request, 'tank_classifier/check_language.html', context)