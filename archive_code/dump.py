

################


# # I can use this to load data with different paths
# data_dir =  '/users/rvinod/data/rvinod/cath/processed_backup'
# paths = os.listdir('/users/rvinod/data/rvinod/cath/processed_backup') #dataset.processed_paths

# #data_list = [torch.load(data_dir+'/'+pt) for pt in paths]



# data_list = []
# for i in range(len(paths)):
#     print(i)
#     with open('readme.txt', 'w') as f:
#         f.write(str(i))
#     data_list.append(torch.load(data_dir+'/'+paths[i]))

# loader = DataLoader(data_list, batch_size=256)

# for step, data in enumerate(loader):
#     print(f'Step {step + 1}:')
#     print('=======')
#     print(f'Number of graphs in the current batch: {data.num_graphs}')
#     print(data)

