import ipywidgets as widgets
from IPython.display import display
from pyspark.ml.pipeline import PipelineModel
loaded_model = PipelineModel.load("./logistic_model_super_final")
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

lb1 = widgets.HTML(value='Chọn NAME_CONTRACT_TYPE:')
lb3 = widgets.HTML(value='Chọn NAME_EDUCATION_TYPE:')
lb4 = widgets.HTML(value='Chọn NAME_TYPE_SUITE:')
lb5 = widgets.HTML(value='Chọn REGION_RATING_CLIENT:')
lb6 = widgets.HTML(value='Nhập AMT_CREDIT:')
lb7 = widgets.HTML(value='Nhập AMT_ANNUAITY:')
lb8 = widgets.HTML(value='Nhập DAYS_BIRTH:')
lb9 = widgets.HTML(value='Nhập EXT_SOURCE_1:')
lb10 = widgets.HTML(value='Nhập EXT_SOURCE_2:')
lb11 = widgets.HTML(value='Nhập EXT_SOURCE_3:')
lb12 = widgets.HTML(value='Nhập AMT_GOODS_PRICE:')
lb13 = widgets.HTML(value='Nhập TOTAL_CLOSED_DEBT:')
lb14 = widgets.HTML(value='Nhập CREDIT_TERM:')
lb15 = widgets.HTML(value='Đang đợi nhập')

lb1.layout.width = '250px'
lb3.layout.width = '250px'
lb4.layout.width = '250px'
lb5.layout.width = '250px'
lb6.layout.width = '250px'
lb7.layout.width = '250px'
lb8.layout.width = '250px'
lb9.layout.width = '250px'
lb10.layout.width = '250px'
lb11.layout.width = '250px'
lb12.layout.width = '250px'
lb13.layout.width = '250px'
lb14.layout.width = '250px'
lb15.layout.max_width = '100%'

NAME_CONTRACT_TYPE = widgets.Dropdown(options=['Cash loans', 'Revolving loans'], disable = False)
CODE_GENDER = widgets.Dropdown(options=['F', 'M','XNA'], disable = False)
NAME_EDUCATION_TYPE = widgets.Dropdown(options=['Secondary / secondary special', 'Higher education','Incomplete higher','Lower secondary','Academic degree'], disable = False)
NAME_TYPE_SUITE = widgets.Dropdown(options=['Unaccompanied', 'Family','Spouse, partner','Children','Other_B','Other_A','Group of people'], disable = False)
REGION_RATING_CLIENT = widgets.Dropdown(options=['1', '2','3'], disable = False)
AMT_CREDIT = widgets.FloatText()
AMT_ANNUAITY = widgets.FloatText()
DAYS_BIRTH = widgets.FloatText()
EXT_SOURCE_1 = widgets.FloatText()
EXT_SOURCE_2 = widgets.FloatText()
EXT_SOURCE_3 = widgets.FloatText()
AMT_GOODS_PRICE = widgets.FloatText()
TOTAL_CLOSED_DEBT = widgets.FloatText()
CREDIT_TERM = widgets.FloatText()

predict_button = widgets.Button(description='Xác nhận')

NAME_CONTRACT_TYPE.layout.width = '200px'
NAME_EDUCATION_TYPE.layout.width = '200px'
NAME_TYPE_SUITE.layout.width = '200px'
REGION_RATING_CLIENT.layout.width = '200px'
AMT_CREDIT.layout.width = '200px'
AMT_ANNUAITY.layout.width = '200px'
DAYS_BIRTH.layout.width = '200px'
EXT_SOURCE_1.layout.width = '200px'
EXT_SOURCE_2.layout.width = '200px'
EXT_SOURCE_3.layout.width = '200px'
AMT_GOODS_PRICE.layout.width = '200px'
TOTAL_CLOSED_DEBT.layout.width = '200px'
CREDIT_TERM.layout.width = '200px'

def predict(b):
    lb15.value = "Đang tính toán..."
    
    # Lấy giá trị từ các widget
    a = NAME_CONTRACT_TYPE.value
    c = NAME_EDUCATION_TYPE.value
    d = NAME_TYPE_SUITE.value
    e = REGION_RATING_CLIENT.value
    f = AMT_CREDIT.value
    g = AMT_ANNUAITY.value
    h = DAYS_BIRTH.value
    i = EXT_SOURCE_1.value
    j = EXT_SOURCE_2.value
    k = EXT_SOURCE_3.value
    l = AMT_GOODS_PRICE.value
    m = TOTAL_CLOSED_DEBT.value
    o = CREDIT_TERM.value
    
    # Tạo DataFrame và chạy dự đoán
    newCus = spark.createDataFrame([
        {'NAME_CONTRACT_TYPE':a,
         'NAME_EDUCATION_TYPE':c,
         'NAME_TYPE_SUITE':d,
         'REGION_RATING_CLIENT':e,
         'AMT_CREDIT':f,
         'AMT_ANNUITY':g,
         'DAYS_BIRTH':h,
         'EXT_SOURCE_1':i,
         'EXT_SOURCE_2':j,
         'EXT_SOURCE_3':k,
         'AMT_GOODS_PRICE':l,
         'TOTAL_CLOSED_DEBT':m,
         'CREDIT_TERM':o}
    ])
    
    # Chạy mô hình
    result = loaded_model.transform(newCus).select("probability").first()[0]
    proba_target = result[1]
    prediction = 'không trả được nợ' if proba_target > 0.200019522474137 else 'trả được nợ'
    print('Mô hình dự đoán: Khách hàng cũ này',prediction,'với tỉ lệ rủi ro là',round(proba_target,3))
    if proba_target > 0.200019522474137 and proba_target < 0.3:
        print('Dù vậy, có thể xem xét lại thông tin khác của KH này để cân nhắc cho họ vay')
    # Cập nhật kết quả
predict_button.on_click(predict)

display(widgets.HBox([lb1, NAME_CONTRACT_TYPE]))
display(widgets.HBox([lb3, NAME_EDUCATION_TYPE]))
display(widgets.HBox([lb4, NAME_TYPE_SUITE]))
display(widgets.HBox([lb5, REGION_RATING_CLIENT]))
display(widgets.HBox([lb6, AMT_CREDIT]))
display(widgets.HBox([lb7, AMT_ANNUAITY]))
display(widgets.HBox([lb8, DAYS_BIRTH]))
display(widgets.HBox([lb9, EXT_SOURCE_1]))
display(widgets.HBox([lb10, EXT_SOURCE_2]))
display(widgets.HBox([lb11, EXT_SOURCE_3]))
display(widgets.HBox([lb12, AMT_GOODS_PRICE]))
display(widgets.HBox([lb13, TOTAL_CLOSED_DEBT]))
display(widgets.HBox([lb14, CREDIT_TERM]))
display(widgets.HBox([predict_button]))


