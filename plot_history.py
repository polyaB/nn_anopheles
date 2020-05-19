import matplotlib.pyplot as plt
# histories look like as [('baseline', baseline_history),
#               ('smaller', smaller_history),
#               ('bigger', bigger_history)]

def plot_history(histories, file_name, key):
    plt.figure(figsize=(16,10))
    for name, history in histories:
        print(name, history.history.keys())
        print('val_'+key)
        val = plt.plot(history.epoch, history.history['val_'+key],
                       '--', label=name.title()+' Val')
        plt.semilogy(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title()+' Train')
    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()
    plt.xlim([0,max(history.epoch)])
    plt.savefig("/mnt/storage/home/psbelokopytova/nn_anopheles/model_graphs/" + file_name +".png", dpi=300)
    plt.close()