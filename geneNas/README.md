# gene-nas
- Module evolution để cho những cái liên quan đến tiến hóa
- Network là để encode từ các gene ra thành 1 mạng RNN
- Module problem là chứa từ data đến define problem đến cách chạy pytorch các thứ
- Các file .py ở ngoài là cách chạy các thực nghiệm (các kịch bản thực nghiệm)

# Yêu cầu:
- Implement lại những module mà người ta sử dụng trong bài NASGEP  ( section III-C, III-D) (https://arxiv.org/abs/2005.07669). Trong code thì các module này sẽ ở cái file function_set.py, tuy nhiên hiện giờ chỉ có các module để tìm RNN thôi.
- 1 gene của mình sẽ có 2 main program, nhiệm vụ của 2 main program là xác định cái architecture của 2 cell normal cell và reduction cell. Sau khi tìm được xong 2 cell đó sẽ nhét vào 1 cái khung định sẵn (như figure 3 trong bài NASGEP) để thành một mạng conv net hoàn chỉnh. Các em sẽ code một model mới lo vụ này (có thể tham khảo code phần recurrent_net.py của a)
- Dựa vào framework pytorch-lightning để code cho phần image classification trên dataset CIFAR-10

# Project sử dụng Frame-work PyTorch Lightning
https://pytorch-lightning.readthedocs.io/en/latest/index.html?fbclid=IwAR3QeExQW_3NhdRZggdaFJSHn87eAmdAnzBAF2wdlmqn-1hCujKpvGjatXY

# Sử dụng toán tử, cấu trúc search của NasGep
- Nasgep: https://arxiv.org/pdf/2005.07669.pdf
- SepConv: https://viblo.asia/p/separable-convolutions-toward-realtime-object-detection-applications-aWj534bpK6m

# Workflow
![alt text](https://github.com/huydang2k/genenas_cv_implement/blob/main/geneNas/image.png)

# Note

Mỗi quần thể gồm nhiều **Chromosome** có dạng:

[ 0  2  8  2  9  4 15 15 16 16 14 14 16 14 15 14 14 15 15  7  5  3  6  0  9 14 15 14 15 16 16 16 14 15 14 16 16 15  8  0  2  5 13 13 13 12 11 12 11 13 12  0  0  2  7 12 13 11 13 12 12 12 13 12]

**D**: chromosome length  
**Popsize**: population size  
**T**: number of iterator  
**Lower_bound/ lb**: List với length = D, chứa cận dưới cho giá trị từng node trong chromosome  
**Upper_bound/ ub**: List với length = D, chứa cận trên cho giá trị từng node trong chromosome  
**Chromosome range**: 4 giá trị đánh đánh dấu các dải giá trị tương ứng của từng loại node (functon, adf, ..) trong chromosome  
Gồm **R1, R2, R3, R4**:  
[0, R1) : function (Được liệt kê trong class function_set.CVFunctionSet, cần chỉnh sửa)  
[R1 ,R2): adf_name. ex a1, a2, a3  
[R2 , R3): adf_terminal_name, ex: t1, t2, t3	  
[R3, R4):  Terminal name (main), ex x1,x2,x3  
Chromosome cho genenas cv sẽ có thêm một main  

Ví dụ với bộ (9, 11,14, 17) và tập function  
{"name": "element_wise_sum", "arity": 2},  
{"name": "element_wise_product", "arity": 2},  
{"name": "concat", "arity": 2},  
{"name": "blending", "arity": 3},  
{"name": "linear", "arity": 1},  
{"name": "sigmoid", "arity": 1},  
{"name": "tanh", "arity": 1},  
{"name": "leaky_relu", "arity": 1},  
{"name": "layer_norm", "arity": 1}  
( **arity** là số lượng tham số tương ứng các hàm)  
Giải mã chromosome:  
[ 0  2  8  2  9  4 15 15 16 16 14 14 16 14 15 14 14 15 15  7  5  3  6  0  9 14 15 14 15 16 16 16 14 15 14 16 16 15  8  0  2  5 13 13 13 12 11 12 11 13 12  0  0  2  7 12 13 11 13 12 12 12 13 12]  
Được  
['element_wise_sum', 'concat', 'layer_norm', 'concat', 'a1', 'linear', 'x2', 'x2', 'x3', 'x3', 'x1', 'x1', 'x3', 'x1', 'x2', 'x1', 'x1', 'x2', 'x2', 'leaky_relu', 'sigmoid', 'blending', 'tanh', 'element_wise_sum', 'a1', 'x1', 'x2', 'x1', 'x2', 'x3', 'x3', 'x3', 'x1', 'x2', 'x1', 'x3', 'x3', 'x2', 'layer_norm', 'element_wise_sum', 'concat', 'sigmoid', 't3', 't3', 't3', 't2', 't1', 't2', 't1', 't3', 't2', 'element_wise_sum', 'element_wise_sum', 'concat', 'leaky_relu', 't2', 't3', 't1', 't3', 't2', 't2', 't2', 't3', 't2']  

**evaluation/optimizer**: có class Optimizer chứa phương thức **ga()**, là luồng thực thi workflow chính  
**evaluation/operator**: class Operator implement các phương thức lai ghép, đột biến. Code thêm
**network/recurrent_net.py**: chứa class **RecurrentNet** định nghĩa module gồm các layers RNN trong mạng genenas nlp, mình sẽ dựa vào đây để code class **NasgepCellNet** trong **Network/nasgep_cell_net.py** để định nghĩa module chứa các layer normal cell, reduction cell  
**problem/lit_recurrent.py**: chứa các class định nghĩa mạng neuron cho bài toán NLP (bao gồm **RecurrentNet** và một số module khác theo mô hình trong paper). Tương tự cho bài toán cv, ta có **problem/nasgetp_net.py  **


