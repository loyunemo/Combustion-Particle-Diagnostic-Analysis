import trackpy as tp

# 加载图像序列
frames = tp.batch('C:/Users/Claudius/WorkingSpace/Article1/0722experiment2/51/BMP192.168.8.51-20240707-072633/*.png', max_frames=10)

# 设置追踪参数
params = {'search_range': (1, 1), 'memory': 1}

# 进行追踪
results = tp.link_df(frames, params)

# 显示追踪结果
tp.plot_traces(results)
