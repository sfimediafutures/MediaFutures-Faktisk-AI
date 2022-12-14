from django import forms


class ImageUploadForm(forms.Form):
    image = forms.ImageField()

class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=50)
    file = forms.FileField()