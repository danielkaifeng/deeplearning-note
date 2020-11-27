
## Tensorflow

### reshape / flatten
```
x = tf.expand_dims(x, -1)
x = tf.layers.flatten(x)

```
### conv2d / maxpooling / dense layer
```
x = tf.layers.conv2d(x, filters=64, kernel_size=(3,3), strides=(2, 2), padding='same', activation=tf.nn.relu)
x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='valid') 
x = tf.layers.batch_normalization(x)
```
### loss 
```
#MSE loss
mse_loss = tf.reduce_mean(tf.squared_difference(self.x_, self.label))

#GAN loss
real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( logits=self.D_real, 
											labels=tf.ones_like(self.D_real))) 

fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( logits=self.D_fake, 
											labels=tf.zeros_like(self.D_fake)))
D_loss = real_loss + fake_loss

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))


```
### optimizer
```
discriminator_vars =  [var for var in tf.global_variables() if  "discriminator" in var.name]
generator_vars =  [var for var in tf.global_variables() if  "discriminator" not in var.name]

self.D_opt = tf.train.AdamOptimizer(learning_rate=d_lr).minimize(self.D_loss, var_list=discriminator_vars)
self.G_opt = tf.train.AdamOptimizer(learning_rate=g_lr).minimize(self.G_loss, var_list=generator_vars)

```
### session

```
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

#or 
with tf.Session(config=config) as sess:
		#do

```

### pixel shuffle
```
def pixel_shuffle(x, block_size=2):
		return tf.depth_to_space(x, block_size=block_size)
```


## Pytorch
### reshape / flatten
```

```

### conv2d / maxpooling / dense layer
```

```
### loss 
```

```
### optimizer
```

```

### pixel shuffle
```
# torch.nn.PixleShuffle(upscale_factor)

ps = nn.PixelShuffle(3)
input = torch.tensor(1, 9, 4, 4)
output = ps(input)

print(output.size())
#torch.Size([1, 1, 12, 12])
```
