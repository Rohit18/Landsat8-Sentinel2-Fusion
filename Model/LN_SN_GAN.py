#Import libraries
import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.gan import *

from PIL import Image, ImageDraw, ImageFont

#Define paths to the training sets. path_l - Landsat 8 (low res), path _s = Sentinel-2 (high res)
path_l = '/LS/Datasets/LN_256/'
path_s = '/LS/Datasets/SN_256/'

#Batch Size dependent on image size, cannot handle larger than 256x256
bs,size=8, 256
arch = models.resnet34

#Get input data
src = ImageImageList.from_folder(path_l).split_by_rand_pct(0.1, seed=42)

def get_data(bs,size):
    data = (src.label_from_func(lambda x: path_s + '/' + x.name)
           .transform(get_transforms(do_flip=True), size=size, tfm_y=True)
           .databunch(bs=bs))
    return data

data_gen = get_data(bs,size)

#Weight Decay, y_range, loss function
wd = 1e-3
y_range = (-3.,3.)

#Define GAN learner, currently selected at not pretrained, pretrained might do better
def create_gen_learner():
    return unet_learner(data_gen, arch, wd=wd, loss_func=F.l1_loss, blur=True, norm_type=NormType.Weight, y_range=y_range, 
        self_attention=True, metrics=[root_mean_squared_error], pretrained=False)

learn_gen = create_gen_learner()

#changed the number of epochs to 120, must check the loss funcitons
learn_gen.fit_one_cycle(120, pct_start=0.8)

#Save the model
learn_gen.save('LN_SN_GEN')

#No need to load again but just trying to be thorough
learn_gen.load('LN_SN_GEN')

#Name the intermediary folder to store the generator predictions
name_gen = 'LN_SN_GAN_Preds'
path_gen = '/users/PAS1437/osu10674/LS/Datasets/LN_SN_GAN_Preds/'

#Check the learn model from where the data is coming
def save_preds(dl):
    i=0
    names = dl.dataset.items
    
    for b in dl:
        preds = learn_gen.pred_batch(batch=b, reconstruct=True)
        for o in preds:
            o.save(path_gen + '/' + names[i].name)
            i += 1

save_preds(data_gen.fix_dl)

#Clear the learn_gen because RAM don't need no too much load
learn_gen=None
gc.collect()

#this is for letting the program access both the folders, real and generated data
path = '/users/PAS1437/osu10674/LS/Datasets/'

#definition for loading critic data, classes are required for the real and fake data
def get_crit_data(classes, bs, size):
    src = ImageList.from_folder(path, include=classes).split_by_rand_pct(0.1, seed=42)
    ll = src.label_from_folder(classes=classes)
    data = (ll.transform(get_transforms(do_flip=True), size=size)
           .databunch(bs=bs))
    return data

#define the data folders, name_gen and the real data again
data_crit = get_crit_data([name_gen, 'SN_256'], bs=bs, size=size)

#define the loss of the loss critic, you know this, you have studied it
loss_critic = AdaptiveLoss(nn.BCEWithLogitsLoss())

#define critic learner
def create_critic_learner(data, metrics):
    return Learner(data, gan_critic(), metrics=metrics, loss_func=loss_critic, wd=wd)

#I am not 100% sure what the accuracy_thresh_expand is
learn_critic = create_critic_learner(data_crit, accuracy_thresh_expand)

#train a bit before the GAN starts training
learn_critic.fit_one_cycle(10, 1e-3)

#Save it sweetheart
learn_critic.save('LN_SN_CRIT')

#Clear your heads, do some meditation
learn_crit=None
learn_gen=None
gc.collect()

#Instantiate the Critic again
data_crit = get_crit_data([name_gen, 'SN_256'], bs=bs, size=size)

#Load critic
learn_crit = create_critic_learner(data_crit, metrics=None).load('LN_SN_CRIT')

#Instantiate the Generate again
learn_gen = create_gen_learner().load('LN_SN_GEN')

#the switcheroo, need to study more
switcher = partial(AdaptiveGANSwitcher, critic_thresh=0.65)
learn = GANLearner.from_learners(learn_gen, learn_crit, weights_gen=(1.,50.), show_img=True, switcher=switcher,
                                 opt_func=partial(optim.Adam, betas=(0.,0.99)), wd=wd)
learn.callback_fns.append(partial(GANDiscriminativeLR, mult_lr=5.))

#How are we setting this without learning the learning rate?
lr = 1e-4

#Learn, 60? or more? 40 was default. 
learn.fit(60,lr)

#save the whole model
learn.save('LN_SN_GAN')

#reduce the learning rate, 10 was default
learn.fit(20,lr/2)

#Save it sweetheart
learn.save('LN_SN_GAN')

#define the new folder for saving the final preds, CHANGE IT!!
path_gen = '/users/PAS1437/osu10674/LS/Datasets/LN_SN_GAN_Final/'

#For predicting all the results from the learned Generator
#Use the learn_gen for prediction
def save_preds_final(dl):
    i=0
    names = dl.dataset.items
    
    for b in dl:
        preds = learn_gen.pred_batch(batch=b, reconstruct=True)
        for o in preds:
            o.save(path_gen + '/' + names[i].name)
            i += 1

#final preds
save_preds_final(data_gen.fix_dl)

#define the new folder for saving the final preds, CHANGE IT!!!
path_gen = '/users/PAS1437/osu10674/LS/Datasets/LN_SN_GAN_Final_Validation/'

save_preds_final(data_gen.valid_dl)


