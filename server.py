from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
import pickle
from PIL import Image
import io
import os
import time
from ultralytics import YOLO

class ImageSearchForK:
    def __init__(self,):
        self.img_lib = {}
        self.classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 
          5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 
          10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 
          14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 
          20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 
          25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 
          30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 
          35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 
          39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 
          45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 
          51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 
          58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 
          64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 
          70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 
          76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

        self.cos_sim = torch.nn.CosineSimilarity(dim=0)

        self.model_embd = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.model_embd = torch.nn.Sequential(*(list(self.model_embd.children())[:-1]))
        self.model_embd.eval()
        print("-:-:-:- Embedding model loaded successfully -:-:-:-")

        self.model_obj_detect = YOLO("yolov10n.pt")
        print("-:-:-:- Object Detection model loaded successfully -:-:-:-")

        self.database_id = 'database_coco_test.pickle'
        if os.path.exists(self.database_id):
            with open(self.database_id, 'rb') as handle:
                self.img_lib = pickle.load(handle)
            print("-:-:-:- Database loaded successfully -:-:-:-")
        else:
            print("-:-:-:- Enable to detect the Database file -:-:-:-")


    def img_embd(self, file_path) -> torch.Tensor:
        print("-:-:-:- Generating embedding adn tags of the input image -:-:-:-")
    
        input_image = Image.open(io.BytesIO(file_path))

        obj_output = self.model_obj_detect(input_image)
        cls = obj_output[0].boxes.cls
        cls = cls.to(torch.int).cpu().tolist()
        cls = list(set(cls))
        tags = ""
        for c in  cls:
            tags += str(self.classes[c]) + ":"

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) 

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.model_embd = self.model_embd.to('cuda')

        with torch.no_grad():
            output = self.model_embd(input_batch)
            output = output.to(torch.float16)
        
        return output.squeeze(), tags
    
    def search_k(self, k, img_pth):
        start = time.time()
        print('-:-:-:- Searching similar images and listing top {} -:-:-:-'.format(k))
        
        temp_indx = {}
        list_tags = []
        ref_img, tags = self.img_embd(img_pth)
        if tags != "NO_TAG":
            list_tags = tags[:-1].split(":")
            list_tags.append(tags)
        else:
            list_tags.append(tags)
            

        for tag in list_tags:
            if tag != tags:
                tag = tag + ":"
            if tag in self.img_lib:
                for key, value in self.img_lib[tag].items():
                    sim_score = self.cos_sim(ref_img, value).cpu().numpy()
                    sim_score =  sim_score.tolist()
                    if len(temp_indx) < k+1:
                        temp_indx[key] = sim_score
                    else:
                        min_key, min_value = min(temp_indx.items(), key=lambda x: x[1])
                        if min_value < sim_score:
                            temp_indx.pop(min_key)
                            temp_indx[key] = sim_score
        
        print('-:-:-:- Total Search Time {} ms'.format(time.time()-start))
        return temp_indx
    
image_search = ImageSearchForK()


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to Image Search Engine"}

@app.post("/search_image/")
async def search_image(file: UploadFile): 
    try:
        file_bytes = await file.read()
        img_name = file.filename
        top_k = img_name[:-4]
        return JSONResponse(content=image_search.search_k(k=int(top_k), img_pth=file_bytes))
    except Exception as e:
        return JSONResponse(content={"error": str(e)})
    