# ComfyUI\_XiNodes

自用节点

## Multi Folder Image Loader

可同时加载最多三个文件夹内的图片，要求每个文件夹内的图片文件名一一对应，以此来进行打标等批量操作。
可输出图像、图像路径、文件名

## Save Text File

保存文本节点，可自定义：保存路径、文件前缀、分隔符、数字padding、后缀、文件格式、编码格式，可被ComfyUI的historyData.outputs捕捉

## Image Batch Crossfade

给两段视频添加叠化过渡效果，可自定义：过渡时长(帧数)，根据批次resize(a/b)，透明通道处理

## Gemini API Node

google-genai协议的api节点，可以用于chat和image，可自定义url,key,model,生图比例、分辨率，最多输入三张图，可输出图像或文本

## Create JSON + Get JSON Value + Format JSON

* Create JSON创建json格式的数据，可以多json串联相加，值可以输入任意格式(string, int, float, boolean, list, json)，int, float, boolean类型用value\_in连接，优先使用连接的数据
* Get JSON Value用于提取json数据的键值，在key\_path输入要提取的键，多层键值用"键值1-键值2-键值3"表示，类似python的\["键值1"]\["键值2"]\[0]，output\_type选择输出的类型
* Format JSON输入标准的json格式的字符串，转换输出为json类型

