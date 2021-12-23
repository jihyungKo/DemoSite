# Design and Planning

###### Ko jihyung

Rev. 1.0 2021.12.21 - initial version(Korean)

*** (Not Yet) Rev. 1.1 2021.12.22 - translated version(English)

### System Architecture

------



![그림1](https://ifh.cc/g/v5opyB.jpg)

![그림2](https://ifh.cc/g/6FJilG.jpg)



### E-R Diagram

------

(수정중)

https://app.diagrams.net/#G1Dvh32TNbkhtqPU3NheSOKRMMXdQyq0I9

### View

------

![페이지1](https://ifh.cc/g/QSOxuu.jpg)

![페이지2](https://ifh.cc/g/ZetAVR.jpg)

![페이지3](https://ifh.cc/g/8XRMGI.jpg)

1. Main page(/main)
   - Show sample input and output on the upper side
   
   - Upload Input, weight and label
   - Until the output is calculated, show wait sign
   - After the inference process, show output
   - When History button clicked, go to History page
2. History page(/hist/all)

   - Show all Log links and Log summaries
   - When Log links are clicked, go to HistoryDetail page
   - When Home button clicked, go to Main page
3. HistoryDetail page(/hist/:id)

   - Show input, weight_name, label and output of each Log
   - When History button clicked, go to History page
   - When Home button clicked, go to Main page

(그림 추가)



### Controller

------

api 와 model, view 간의 관계 명시



### Design Details

###### Frontend

------

< Container > 

1. Main
   - componentDidMount()
   - onClickHistory()
2. History
   - onClickMain()
3. HistoryDetail
   - onClickHistory()
   - onClickMain()

< Component >

1. buttons

2. informations
   - LogDetailInfo, LogSimpleInfo

< Store >

< Reducer >

- Reducer

###### API

------

- Reducer

| API          |     GET      |     POST      | PUT  | DELETE | PATCH |
| ------------ | :----------: | :-----------: | :--: | :----: | :---: |
| api/input    |      X       | UPLOAD_INPUT  |  X   |   X    |   X   |
| api/label    |      X       | UPLOAD_LABEL  |  X   |   X    |   X   |
| api/weight   |      X       | UPLOAD_WEIGHT |  X   |   X    |   X   |
| api/hist/all | GET_ALL_HIST |       X       |  X   |   X    |   X   |
| api/hist/:id |   GET_HIST   |       X       |  X   |   X    |   X   |
| api/output   |  GET_OUTPUT  |       X       |  X   |   X    |   X   |



###### Backend

------

model에 대한 정보는 미리 담아둔다



### For Inference

------

Deploying Pytorch in python via a rest API with flask

https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html



### Docker & Kubernetes

------

진행하면서 추가