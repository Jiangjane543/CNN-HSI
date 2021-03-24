import Hyperspectral_input

DATA_FLAG = 0

#indian_pines data 0
indianP_origin_data = Hyperspectral_input.input_data('./Hyperspectral Image datasets/Indian_pines_corrected.mat',
                                                     'indian_pines_corrected')
indianP_gt = Hyperspectral_input.input_data('./Hyperspectral Image datasets/Indian_pines_gt.mat',
                                            'indian_pines_gt')
'''
# Pavia University data 1
paviaU_origin_data = Hyperspectral_input.input_data('./Hyperspectral Image datasets/PaviaU.mat', 'paviaU')
paviaU_gt = Hyperspectral_input.input_data('./Hyperspectral Image datasets/PaviaU_gt.mat', 'paviaU_gt')
# Salinas data 2
Salinas_origin_data = Hyperspectral_input.input_data('./Hyperspectral Image datasets/Salinas_corrected.mat',
                                                     'salinas_corrected')
Salinas_gt = Hyperspectral_input.input_data('./Hyperspectral Image datasets/Salinas_gt.mat', 'salinas_gt')
# contest2013 data 3
contest2013_origin_data = Hyperspectral_input.input_data('./Hyperspectral Image datasets/contest2013.mat',
                                                         'contest2013')
contest2013_gt = Hyperspectral_input.input_data('./Hyperspectral Image datasets/contest2013_gt.mat',
                                                'contest2013_gt')
# contest2014 data 4
contest2014_origin_data = Hyperspectral_input.input_data('./Hyperspectral Image datasets/contest2014.mat',
                                                         'contest2014')
contest2014_gt = Hyperspectral_input.input_data('./Hyperspectral Image datasets/contest2014_gt.mat',
                                                'contest2014_gt')
# paviaC data 5
paviaC_origin_data = Hyperspectral_input.input_data('./Hyperspectral Image datasets/Pavia.mat', 'pavia')
paviaC_gt = Hyperspectral_input.input_data('./Hyperspectral Image datasets/Pavia_gt.mat', 'pavia_gt')
'''

if DATA_FLAG == 0:
    hyperdate = Hyperspectral_input.Hyperimage(indianP_origin_data, indianP_gt)
    gt_data = indianP_gt
'''
elif DATA_FLAG == 1:
    hyperdate = Hyperspectral_input.Hyperimage(paviaU_origin_data, paviaU_gt)
    gt_data = paviaU_gt
elif DATA_FLAG == 2:
    hyperdate = Hyperspectral_input.Hyperimage(Salinas_origin_data, Salinas_gt)
    gt_data = Salinas_gt
elif DATA_FLAG == 3:
    hyperdate = Hyperspectral_input.Hyperimage(contest2013_origin_data, contest2013_gt)
    gt_data = contest2013_gt
elif DATA_FLAG == 4:
    hyperdate = Hyperspectral_input.Hyperimage(contest2014_origin_data, contest2014_gt)
    gt_data = contest2014_gt
elif DATA_FLAG == 5:
    hyperdate = Hyperspectral_input.Hyperimage(paviaC_origin_data, paviaC_gt)
    gt_data = paviaC_gt
'''