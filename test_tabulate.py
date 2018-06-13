print('[Epoch {:d}]             |   (training accuracy %)   |   (testing accuracy %)   |'.format(0))
print('[CNN+RN]  Loss: {:5.2f} | rel: {:5.2f}, nonrel: {:5.2f} | rel: {:5.2f}, nonrel: {:5.2f}|'.format(12.2345, 1,2,3,4,5))
print('[CNN+MLP] Loss: {:5.2f} | rel: {:5.2f}, nonrel: {:5.2f} | rel: {:5.2f}, nonrel: {:5.2f}|'.format(12.2345, 1,2,3,4,5))




import time

start = time.time()

time.sleep(2)

delta_t = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))

print(delta_t)