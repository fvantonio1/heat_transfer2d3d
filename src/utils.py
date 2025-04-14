from sklearn.preprocessing import MinMaxScaler

def scale_data(data, scalers=None, scale_temp=False):

    if not scalers:
        scalers = []

        i = data.shape[1]
        i -= 1 if not scale_temp else 0

        for j in range(i):
            scaler = MinMaxScaler()
            data[:, j] = scaler.fit_transform(data[:, j].reshape(-1, 1)).reshape(-1)
            scalers.append(scaler)

        return data, scalers
    
    for j in range(len(scalers)):
        data[:, j] = scalers[j].transform(data[:, j].reshape(-1, 1)).reshape(-1)

    return data

