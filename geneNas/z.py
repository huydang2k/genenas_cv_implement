
# import numpy as np
# import pickle
# with open('/hdd/huydang/genenas_cv_implement/geneNas/lightning_logs/version_45/events.out.tfevents.1641229890.ubuntu.15678.0','rb') as f:
#         t = pickle.load(f)
# print(t)
import pickle
with open('logs/chromosome19.500epoch.2e5.06_01_2022.uniform.log_infor.pkl','rb') as f:
        d = pickle.load(f)
print(d)
# print(d['acc'][str(20)])       
# # m = -1
# i_m = -1        
# for i in range(20):
#         l = -1
#         i_l = -1
#         for j in range(100):
#                 if d['acc'][str(i+ 1)][j] > l:
#                         l = d['acc'][str(i + 1)][j]
#                         i_l = j
#         print(f'index {i} -- max {l} on {i_l}')
 
#         if (l > m):
#                 m = l
#                 i_m = i
# print('over all--')
# print(m)
# print(i_m)


