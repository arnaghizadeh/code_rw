import torch
import os, sys
import numpy as np
import argparse
import cv2
import time
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # this is for the main directory

from xlwt import Workbook
from PIL import Image
import re
import copy
import skimage.transform
import skimage
from skimage.measure import label as sklabel
from skimage import morphology

import InstSeg.P as P
import InstSeg.KGnet as KGnet
import InstSeg.postprocessing as postprocessing
import InstSeg.nms as nms
from InstSeg.nosubdataset_16bits import ImageNikonND2

from PIX2PIX.util import util


def parse_args():
    parser = argparse.ArgumentParser(description="InstanceHeat")
    parser.add_argument("--data_dir", help="data directory", default=P.data_dir, type=str)
    parser.add_argument("--pretrained_weights", help="data directory", default=P.pretrained_weights, type=str)
    parser.add_argument("--postrained_weights", help="data directory", default=P.postrained_weights, type=str)
    #parser.add_argument("--source_dir", help="source directory", default=P.source_dir, type=str)
    parser.add_argument("--resume", help="resume file", default="end_model.pth", type=str)
    parser.add_argument('--input_h', type=int, default=512, help='input height')
    parser.add_argument('--input_w', type=int, default=512, help='input width')
    parser.add_argument('--save_img', type=bool, default=True, help='save img or not')
    parser.add_argument('--nms_thresh', type=float, default=0.3, help='nms_thresh')
    parser.add_argument('--seg_thresh', type=float, default=0.5, help='seg_thresh')
    parser.add_argument("--dataset", help="training dataset", default='CustomDS', type=str)
    args = parser.parse_args()
    return args


class InstanceHeat(object):
    def __init__(self):
        self.model = KGnet.resnet50(pretrained=True)
        self.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    def data_parallel(self):
        self.model = torch.nn.DataParallel(self.model)

    def load_weights(self, resume, args):
        self.model.load_state_dict(torch.load(os.path.join(args.postrained_weights, resume)))

    def map_mask_to_image(self, mask, img, color):
        # color = np.random.rand(3)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mskd = img * mask
        clmsk = np.ones(mask.shape) * mask
        clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
        clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
        clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
        img = img + 1. * clmsk - 1. * mskd
        return np.uint8(img)

    def show_heat_mask(self, mask):
        mask = mask - np.min(mask)
        mask = mask / np.max(mask)
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        return heatmap

    def imshow_kp(self, kp, img_in):
        h, w = kp.shape[2:]
        img = cv2.resize(img_in, (w, h))
        colors = [(0, 0, 0.9), (0.9, 0, 0), (0.9, 0, 0.9), (0.9, 0.9, 0), (0.2, 0.9, 0.9)]
        for i in range(kp.shape[1]):
            img = self.map_mask_to_image(kp[0, i, :, :], img, color=colors[i])
        return img

    def test_inference(self, args, image, bbox_flag=False):
        image = np.dstack([image, image, image])
        height, width, c = image.shape  #

        img_input = cv2.resize(image, (args.input_w, args.input_h))
        img_input = torch.FloatTensor(np.transpose(img_input.copy(), (2, 0, 1))).unsqueeze(0) / 255 - 0.5
        img_input = img_input.to(self.device)

        with torch.no_grad():
            begin = time.time()
            pr_c0, pr_c1, pr_c2, pr_c3, feat_seg = self.model.forward_dec(img_input)
            print("forward time is {:.4f}".format(time.time() - begin))
            pr_kp0, pr_short0, pr_mid0 = pr_c0
            pr_kp1, pr_short1, pr_mid1 = pr_c1
            pr_kp2, pr_short2, pr_mid2 = pr_c2
            pr_kp3, pr_short3, pr_mid3 = pr_c3

        torch.cuda.synchronize()
        skeletons0 = postprocessing.get_skeletons_and_masks(pr_kp0, pr_short0, pr_mid0)
        skeletons1 = postprocessing.get_skeletons_and_masks(pr_kp1, pr_short1, pr_mid1)
        skeletons2 = postprocessing.get_skeletons_and_masks(pr_kp2, pr_short2, pr_mid2)
        skeletons3 = postprocessing.get_skeletons_and_masks(pr_kp3, pr_short3, pr_mid3)

        skeletons0 = postprocessing.refine_skeleton(skeletons0)
        skeletons1 = postprocessing.refine_skeleton(skeletons1)
        skeletons2 = postprocessing.refine_skeleton(skeletons2)
        skeletons3 = postprocessing.refine_skeleton(skeletons3)

        bboxes = postprocessing.gather_skeleton(skeletons0, skeletons1, skeletons2, skeletons3)
        bboxes = nms.non_maximum_suppression_numpy(bboxes, nms_thresh=0.5)
        if bbox_flag:
            return bboxes
        if bboxes is None:
            return None

        with torch.no_grad():
            predictions = self.model.forward_seg(feat_seg, [bboxes])
        predictions = self.post_processing(args, predictions, width, height)
        return predictions

    def post_processing(self, args, predictions, image_w, image_h):
        if predictions is None:
            return predictions
        out_masks = []
        out_dets = []
        mask_patches, mask_dets = predictions
        for mask_b_patches, mask_b_dets in zip(mask_patches, mask_dets):
            for mask_n_patch, mask_n_det in zip(mask_b_patches, mask_b_dets):
                mask_patch = mask_n_patch.data.cpu().numpy()
                mask_det = mask_n_det.data.cpu().numpy()
                y1, x1, y2, x2, conf = mask_det
                y1 = np.maximum(0, np.int32(np.round(y1)))
                x1 = np.maximum(0, np.int32(np.round(x1)))
                y2 = np.minimum(np.int32(np.round(y2)), args.input_h - 1)
                x2 = np.minimum(np.int32(np.round(x2)), args.input_w - 1)

                mask = np.zeros((args.input_h, args.input_w), dtype=np.float32)
                mask_patch = cv2.resize(mask_patch, (x2 - x1, y2 - y1))

                mask[y1:y2, x1:x2] = mask_patch
                mask = cv2.resize(mask, (image_w, image_h))
                mask = np.where(mask >= 0.5, 1, 0)

                y1 = float(y1) / args.input_h * image_h
                x1 = float(x1) / args.input_w * image_w
                y2 = float(y2) / args.input_h * image_h
                x2 = float(x2) / args.input_w * image_w

                out_masks.append(mask)
                out_dets.append([y1, x1, y2, x2, conf])
        return [np.asarray(out_masks, np.float32), np.asarray(out_dets, np.float32)]

    def imshow_instance_segmentation(self, masks,  img_ids, out_imgs, sheet1, sheet2, sheet3, count, imgs16, selected_idx, compare_str, path):
        single_cnt = 0
        for mask in masks:
            color = np.random.rand(3)
            count = count + 1
            single_cnt = single_cnt + 1

            sheet1.write(count, 0, compare_str)
            sheet2.write(count, 0, compare_str)
            sheet3.write(count, 0, compare_str)
            sheet1.write(count, 1, single_cnt)
            sheet2.write(count, 1, single_cnt)
            sheet3.write(count, 1, single_cnt)



            for c in range(imgs16.shape[0] - 1):  #
                imgs_c = imgs16[c, :, :]
                #mask = np.zeros((np.shape(imgs_c)))
                #mask = cv2.drawContours(mask, contours, -1, 1, -1)

                # mask = np.stack((mask, mask, mask), axis=-1)

                total_intensity = np.sum(imgs_c * mask)
                area = mask.sum()
                mean_intensity = total_intensity / area
                sheet1.write(count, c + 2, total_intensity)
                sheet2.write(count, c + 2, area)
                sheet3.write(count, c + 2, mean_intensity)

        return sheet1, sheet2, sheet3, count


    def find_max_index(self, imgs, channel=2):
        max_value = -1000
        max_index = -1
        for i in range(imgs.shape[1]):
            # print(imgs[channel, i, :, :].mean())
            if imgs[channel, i, :, :].mean() > max_value:
                max_value = imgs[channel, i, :, :].mean()
                max_index = i
        return max_index

    def or_labels(self, labels1,labels2):
        list1 = np.unique(labels1)
        list2 = np.unique(labels2)
        lst = [0]
        labels = np.zeros_like(labels1)
        lbl = 1
        for l in list1:
            if l == 0:
                continue
            labels[labels1==l] = lbl
            lst.append(lbl)
            lbl+=1
        for l in list2:
            if l == 0:
                continue
            labels[labels2==l] = lbl
            lst.append(lbl)
            lbl+=1
        return labels,lst
    def get_pr_masks_jr(self, pr_masks_jr):
        mask_all = np.zeros((1024,1024))
        lbl = 1
        for x in pr_masks_jr:
            mask_all[x>0] = lbl
            lbl += 1
        return mask_all, lbl
    def make_labels(self, img2, pr_masks_jr, lbl):
        for i in range(lbl):
            if i ==0:
                continue
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = np.zeros_like(img2)
            mask[pr_masks_jr==i] = 100
            mask = cv2.dilate(mask, kernel, iterations=5)
            img2[mask > 0] = 0
        img2[img2 <= 60] = 0
        img2[img2 > 0] = 255
        img2 = sklabel(img2)
        img2 = skimage.morphology.remove_small_objects(img2, min_size=64, connectivity=1, in_place=False)
        img2 = np.uint8(img2)
        #ret, img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  #
        img2,lst = self.or_labels(pr_masks_jr,img2)
        return img2,lst

    def test_method4(self, args, name,cell2index,compare,dsets,p2p):
        from torchvision import transforms
        from skimage.transform import resize, rescale
        import torchvision.models as models
        resume_cls = '/research/cbim/vast/an499/Datasets/AugmentJournal/Raw Data - TIFF/checkpoint.pth.tar'#'/mnt/Volume D/exctracted_images/Raw Data - TIFF/checkpoint.pth.tar'  # "D:\\exctracted_images\\Raw Data - TIFF\\checkpoint.pth.tar"
        #checkpoint_cls = torch.load(resume_cls)
        model_cls = models.__dict__['resnet18'](num_classes=3)
        #model_cls.load_state_dict(checkpoint_cls['state_dict'])
        model_cls.eval()
        device = torch.device("cuda:3")  # cuda:0 # cpu
        model_cls.to(device)
        self.load_weights(resume=args.resume, args=args)
        self.model = self.model.to(self.device)
        self.model.eval()

        # if not os.path.exists("save_result") and save_flag is True:
        #    os.mkdir("save_result")

        # dsets = CustomDS(data_dir=args.data_dir, phase='test')
        count = 0

        wb = Workbook()
        sheet1 = wb.add_sheet('Total intensity')
        sheet1.write(0, 0, 'tif file')
        sheet1.write(0, 1, 'cell number')
        sheet1.write(0, 2, 'channel0-F-actin')
        sheet1.write(0, 3, 'channel1-Perforin')
        sheet1.write(0, 4, 'channel2-Tumor Antigen')
        sheet1.write(0, 5, 'channel3-pZeta')

        sheet2 = wb.add_sheet('Area')
        sheet2.write(0, 0, 'tif file')
        sheet2.write(0, 1, 'cell number')
        sheet2.write(0, 2, 'channel0-F-actin')
        sheet2.write(0, 3, 'channel1-Perforin')
        sheet2.write(0, 4, 'channel2-Tumor Antigen')
        sheet2.write(0, 5, 'channel3-pZeta')

        sheet3 = wb.add_sheet('Mean intensity')
        sheet3.write(0, 0, 'tif file')
        sheet3.write(0, 1, 'cell number')
        sheet3.write(0, 2, 'channel0-F-actin')
        sheet3.write(0, 3, 'channel1-Perforin')
        sheet3.write(0, 4, 'channel2-Tumor Antigen')
        sheet3.write(0, 5, 'channel3-pZeta')

        path = os.path.join(name, name1+name2)
        path_plots = os.path.join(name, name1+name2+'plots')
        if not os.path.exists(name):
            os.mkdir(name)
        if not os.path.exists(path):
            os.mkdir(path)
        if not os.path.exists(path_plots):
            os.mkdir(path_plots)

        # initialize

        A1 = 0
        A2 = 0
        for index_dir in range(len(dsets)):  # go into folders
            imgs8 = dsets.getitem_with_ub(index_dir)  # 5x5x1024x1024
            imgs16 = dsets.getitem_no_ub(index_dir)
            index_z = self.find_max_index(imgs16, channel=cell2index["antigen"])  # find the most antigen
            print("z:", index_z, np.shape(imgs8), np.shape(imgs16), index_dir)
            #img_id = dsets1.image_paths[index_dir][0][index_z]
            img_id = dsets.image_paths[index_dir]
            if re.search("Patient 3", img_id) is not None:
                print("Patient 3:",img_id)
                compare_str = "A1"
            else:
                print("Patient 4:",img_id)
                compare_str = "A2"

            num_ch = np.shape(imgs8)[0]
            imgs8 = imgs8[:, index_z, :, :]
            imgs16 = imgs16[:, index_z, :, :]


            out_imgs = []
            img_ids = []
            for ch in range(num_ch):
                out_img = np.stack((imgs8[ch], imgs8[ch], imgs8[ch]), axis=-1)
                tmp = img_id+"_"+str(ch)
                out_imgs.append(out_img)
                img_ids.append(tmp)
            img8_0 = imgs8[0]

            """"""
            imgs8_0 = p2p.img2blocks(img8_0, combine=True)
            
            p2p_0 = p2p.test(imgs8_0)
            p2p_0= p2p.blocks2img(p2p_0)
            p2p_0 = cv2.cvtColor(p2p_0, cv2.COLOR_BGR2GRAY)
            #p2p_0[p2p_0 > 0] = 1

            #p2p_0 = np.zeros((1024,1024))

            jr_0_mask = np.zeros((1024, 1024))
            jr_0_masks = self.test_inference(args, img8_0)            
            if jr_0_masks is not None:
                jr_0_masks, _ = jr_0_masks
                for m in jr_0_masks:
                    jr_0_mask+=m
                jr_0_mask[jr_0_mask>0] = 1

            pr_masks_jr, lbl = self.get_pr_masks_jr(jr_0_masks)
            lbl = 1
            pr_masks_jr = np.zeros((1024, 1024))
            markers, _ = self.make_labels(copy.deepcopy(p2p_0), pr_masks_jr, lbl)
            #jr_0_mask = np.zeros((1024, 1024))
            print(np.shape(jr_0_mask),np.shape(p2p_0))
            pr_masks = []
            ann_ids = np.unique(markers)
            for idx, i in enumerate(ann_ids):
                if idx == 0:
                    continue
                mask = copy.deepcopy(markers)
                mask = np.uint8(mask)
                mask[mask != i] = 0
                mask[mask == i] = 255

                mask_tmp = np.zeros((1024, 1024))
                mask_tmp[mask > 0] = 1
                pr_masks.append(mask_tmp)

            #shownp(p2p_0*100)
            #shownp(jr_0_mask*100)


            mask_sum = np.sum(jr_0_mask) + np.sum(p2p_0)
            if mask_sum == 0:
                continue
            sheet1, sheet2, sheet3, count = self.imshow_instance_segmentation(pr_masks, img_ids,
                                                                              out_imgs, sheet1, sheet2, sheet3, count,
                                                                              imgs16, index_z, compare_str, path)#,len(jr_0_masks))
            print("Image",index_dir," finished out of", len(dsets))
            #sys.exit()
        wb.save(name + '/' + name1 + "_" + name2+ "_method4.xls")
        """
        plots = ['Mean intensity', 'Area', 'Total intensity']
        for plot in plots:
            path1 = name + '/' + name1 + "_" + name2+ ".xls"
            path2 = path_plots + '/' + name1 + "_" + name2 + " "+ plot +".png"
            intensities_visualize.plot(plot,path1,path2)"""

def shownp(arr):
    Image.fromarray(arr).show()

class pix2pix():
    def __init__(self):
        print("Class created!")
    def combineImgs(self, img1, img2):
        imgh = np.concatenate((img1, img2), axis=1)#stack horizontally
        return imgh

    def img2blocks(self, img,nRows = 4,mCols = 4,combine=True):
        # Dimensions of the image
        sizeX = img.shape[1]
        sizeY = img.shape[0]
        imgs = []
        for i in range(0, nRows):
            for j in range(0, mCols):
                roi = img[int(i * sizeY / nRows):int(i * sizeY / nRows + sizeY / nRows),
                      int(j * sizeX / mCols):int(j * sizeX / mCols + sizeX / mCols)]
                if combine:
                    roi = self.combineImgs(roi,roi)
                    roi = Image.fromarray(roi).convert("RGB")
                imgs.append(roi)
        return imgs
    def blocks2img(self, imgs, nRows = 4,mCols = 4):
        id = 0
        imgh = None
        for i in range(0,nRows):
            imgv = None
            for j in range(0, mCols):
                img2 = imgs[id]#readimg(path+'/'+ str(id) + ".png")
                #img2 = cv2.imread(path+'/'+ str(id) + ".png")
                #img2 = instance_segment(img2)
                #img2 = cv2.imread('patches/'+'img'+str(id)+'_'+str(i)+str(j)+".png")
                if imgv is not None:
                    imgv = np.concatenate((imgv, img2), axis=1)#stack horizontally
                else:
                    imgv = img2
                id += 1
            if imgh is not None:
                imgh = np.concatenate((imgh, imgv), axis=0)#stack vertically
            else:
                imgh = imgv
        return imgh

    def test(self,imgs):
        from PIX2PIX.options.test_options import TestOptions
        from PIX2PIX.data import create_dataset
        from PIX2PIX.models import create_model

        opt = TestOptions().parse()  # get test options
        # hard-code some parameters for test
        opt.num_threads = 0  # test code only supports num_threads = 0
        opt.batch_size = 1  # test code only supports batch_size = 1
        opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
        opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
        #img = Image.open("/research/cbim/vast/an499/Datasets/Buttom-Up/200_cart_color/test/0.png").convert('RGB')
        #imgs = [img]

        dataset = create_dataset(opt,imgs)  # create a dataset given opt.dataset_mode and other options
        model = create_model(opt)  # create a model given opt.model and other options
        model.setup(opt)  # regular setup: load and print networks; create schedulers
        if opt.eval:
            model.eval()
        fakes = []
        for i, data in enumerate(dataset):
            print(type(data))
            model.set_input(data)  # unpack data from data loader
            model.test()  # run inference
            visuals = model.get_current_visuals()  # get image results
            for label, im_data in visuals.items():
                if label=="fake_B":
                    im = util.tensor2im(im_data)
                    fakes.append(im)
                    #sys.exit()
        """img = self.blocks2img(fakes)
        img[img>3] = 255
        shownp(img)
        sys.exit()"""
        return fakes


if __name__ == '__main__':
    p2p = pix2pix()
    """
    p2p = pix2pix()
    img = Image.open("/research/cbim/vast/an499/Datasets/Buttom-Up/cart_fake/test/AF647-Streptavidin-Protein A AF488-Perforin AF568-P-Zeta AF405-F-actin-3.png").convert('RGB')
    img = np.array(img)
    imgs = p2p.img2blocks(img, combine=True)    
    p2p_imgs = p2p.test(imgs)
    sys.exit()"""
    f = "/research/cbim/vast/an499/Datasets/AugmentJournal/"
    cell2index = {"factin": 0, "perforin": 1, "antigen": 2, "pZeta": 3, "DIC": 4}

    compare_type = [("Patient 3","Patient 4")]

    args = parse_args()
    object_is = InstanceHeat()
    for val in compare_type:
        compare = {0: val[0], 1: val[1]}
        name = "nd2Houston"
        name1 = compare[0]
        name2 = compare[1]
        f1 = os.path.join(f, name)
        dsets = ImageNikonND2(f1)

        object_is.test_method4(args,name,cell2index,compare,dsets,p2p)
