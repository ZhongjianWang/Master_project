from flask import Flask, request, jsonify
import random
import numpy as np


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
@app.route('/',methods=["POST"])
def hello():
    if request.method == 'POST':
        data = request.form['data']
        print(data)
        
        #segments = np.empty((90,3))
        #qq = data.split(';')
        #for i in range(90):
        #     aa = list(map(float, qq[i].split(',')))
        #     segments = np.dstack([segments, aa])
        # # (180, 3)
        # print(segments)
        # print(segments.shape)
        # segments = np.empty((90,3))

        # session =  tf.compat.v1.Session()
        # saver = tf.compat.v1.train.import_meta_graph('C:/Users/TUL/Desktop/CNN/model/model.meta')
        # saver.restore(session, tf.train.latest_checkpoint('C:/Users/TUL/Desktop/CNN/model/'))
        # tt = session.run(tf.compat.v1.get_default_graph().get_tensor_by_name("prediction:0"), feed_dict={tf.compat.v1.get_default_graph().get_tensor_by_name("X:0"): reshaped_realdata})
        # print(tt)
        # print("Downstairs:{:.2f}, Jogging:{:.2f}, Sitting:{:.2f}, Standing:{:.2f}, Upstrairs:{:.2f}, Working:{:.2f}".format(tt[0][0], tt[0][1], tt[0][2], tt[0][3], tt[0][4] ,tt[0][5]))  
        # session.close()
        
        # 以下是模拟数据，这里替换为处理后的数据
        a = random.randint(0, 100)
        b = random.randint(0, 100)
        c = random.randint(0, 100)
        recognize_info = {'walk': a / 100, 'jump': b / 100, 'run': c / 100}
        return jsonify(recognize_info), 200

    # while True:
    #     a = random.randint(0, 100)
    #     b = random.randint(0, 100)
    #     c = random.randint(0, 100)
    #     if a + b + c == 100 and a!=0 and b!=0 and c!=0:
    #         recognize_info = {'walk': a/100, 'jump': b/100, 'run': c/100}
    #         return jsonify(recognize_info), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0')
    