File Types and Classification
## Images (e.g., .png, .jpg):

**Types:** 
- Screenshot_2024-05-26_114815.png
- android-chrome-512x512.png
- logo-removebg-preview_1.png
- miles-and-gwen-fanart-fe.jpg
- 4cc9731f193638ed18404269d34ad218.jpg
- 14358.jpg

**Classification Goal:** 
If you’re classifying images, you might want to check what category these images fall into (e.g., "Logo", "Screenshot", "Fan Art", etc.).

## PDF Documents (e.g., .pdf):

**Types:** 
- KRISHOS-REST.pdf
- PFSD_SET_2.pdf
- invoice.pdf
- Machine_Learning_Notes.pdf

**Classification Goal:** 
If you’re classifying text within PDFs, you might want to categorize these documents (e.g., "Machine Learning Notes", "Invoice", "Research Paper", etc.).

## Text File (e.g., .txt):

**Types:** 
- keypair_1.txt

**Classification Goal:** 
For text files, you might want to categorize the contents (e.g., "Configuration", "Keypair", etc.).

## JSON File (e.g., .json):

**Types:** 
- vulnerabilities.json

**Classification Goal:** 
If this JSON file contains vulnerability data, you might classify the types of vulnerabilities or the importance of each entry.

## Steps to Test Classification

### For PDFs and Text Files:

1. Extract Text: Use a tool or script to extract text from these files.
2. Classify Extracted Text: Send the extracted text to your classification model. For example, if you're using a POST request, your JSON payload might look like this:

```json
{
    "text": "Extracted text from KRISHOS-REST.pdf"
}
```

**Expected Output:** 
The response might indicate a category like "Research Paper" or "Invoice."

### For Images:

1. Preprocess Images: Ensure your model can handle image input. You might need to use an image classification model or convert the images into a format your model accepts.
2. Classify Images: Send the image data to your classification model. This might involve converting the image to base64 and sending it in a POST request:

```json
{
    "image": "base64_encoded_image_data"
}
```

**Expected Output:** 
The response might classify the image into categories like "Logo" or "Screenshot."

### For JSON Files:

1. Analyze JSON Content: Understand the structure and content of the JSON file.
2. Classify Based on Content: You might send relevant parts of the JSON file (e.g., descriptions or key details) to the classification model:

```json
{
    "text": "Content extracted from vulnerabilities.json"
}
```

**Expected Output:** 
The classification result might identify categories related to vulnerabilities or their impact.