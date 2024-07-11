import cv2
import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt
from skimage.transform import PiecewiseAffineTransform, PolynomialTransform, AffineTransform, EuclideanTransform,warp, SimilarityTransform

from skimage.feature import (corner_harris, corner_subpix, corner_peaks,
                             plot_matches)
from skimage.transform import warp, AffineTransform
from skimage.exposure import rescale_intensity
from skimage.color import rgb2gray
from skimage.measure import ransac
from skimage.metrics import mean_squared_error
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim


import random
import math
f = open("stim_presentation.txt", "r")
i = 0
header = []
trials = []
for trial in f:
    if (i == 0):
        header = trial.split()
    else:
        trials.append(trial.split())
    i = i + 1

tu = [] #texture uniform
tr = [] #texture rule
su = [] #symbolic uniform
sr = [] #symbolic rule

for t in trials:
    condition = t[2]
    if (condition == '1'):
        tu.append(t)
    elif (condition == '2'):
        tr.append(t)
    elif (condition == '3'):
        su.append(t)
    else:
        sr.append(t)
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )      
class ShapeDetector:
	def __init__(self):
		pass
	def detect(self, c):
		# initialize the shape name and approximate the contour
		shape = "unidentified"
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.04 * peri, True)
		# if the shape is a triangle, it will have 3 vertices
		if len(approx) == 3:
			shape = "triangle"
		# if the shape has 4 vertices, it is either a square or
		# a rectangle
		elif len(approx) == 4 and cv2.isContourConvex(approx):
			approx = approx.reshape(-1, 2)
			max_cos = np.max([angle_cos( approx[i], approx[(i+1) % 4], approx[(i+2) % 4] ) for i in range(4)])
			if max_cos < 0.1:
			# compute the bounding box of the contour and use the
			# bounding box to compute the aspect ratio
			#(x, y, w, h) = cv2.boundingRect(approx)
			#ar = w / float(h)
			# a square will have an aspect ratio that is approximately
			# equal to one, otherwise, the shape is a rectangle
			#shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
				shape = "rectangle"
		# if the shape is a pentagon, it will have 5 vertices
		elif len(approx) == 5:
			shape = "pentagon"
		# otherwise, we assume the shape is a circle
		else:
			shape = "circle"
		# return the name of the shape
		return shape

def returnNewTransform1(src,dst):
    #try:
    tform, inliers = ransac((src, dst), SimilarityTransform, min_samples=3,
                                               residual_threshold=2, max_trials=5000)
    outliers = (inliers == False)
    return (tform,outliers,"aff")

def applyRule(TFORM,Filter,image):
    out = 0
    for i in range(len(TFORM)):
        step = warp(image, TFORM[i],cval=1, mode ='constant')
        out = out + step
        out = np.array(out, dtype = np.uint8)
        blur = cv2.GaussianBlur(out,(3,3),1)
        _,out = cv2.threshold(blur,out.max()*Filter[i],out.max(),cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    out = np.array(out, dtype = np.float64)
    out *= (255.0/out.max())
    out = np.array(out, dtype = np.uint8)
    return out

def returnNewTransform2(src,dst,sample_size):
    try:
        tform, inliers = ransac((src, dst), AffineTransform, min_samples=sample_size,
                                               residual_threshold=2, max_trials=1000)
        outliers = (inliers == False)
        return (tform,outliers,"aff")
    except:
        return (None, None,"")
    
#--- COST FUNCTION ------------------------------------------------------------+
def checkSimilarityScore(TFORM,Filter,BoxList):
    out = 0
    for i in range(len(TFORM)):
        step = warp(BoxList[1], TFORM[i],cval=1, mode ='constant')
        out = out + step
        #out *= (255.0/out.max())
        #out *= 255.0
        out = np.array(out, dtype = np.uint8)
        blur = cv2.GaussianBlur(out,(3,3),1)
        _,out = cv2.threshold(blur,Filter[i]*out.max(),out.max(),cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    out = np.array(out, dtype = np.float64)
    
    out *= (255.0/out.max())
    out = np.array(out, dtype = np.uint8)
    #ssim_const = ssim(BoxList[2], out,
    #                  data_range=out.max() - out.min())
    ssim_const = mse(BoxList[2], out)
    return ssim_const

# function we are attempting to optimize (minimize)
def Function2Optimize(x,inlier_idxs,outlier_idxs,src,dst,BoxList,TFORM,Filter):#x = set(x)
   # t = x[3]
   # m=np.linspace(round(x[0]), round(x[1]), num=round(x[2]),dtype=int) 
   # m = inlier_idxs[m]
    m =np.linspace(round(x[0]), round(x[1]), num=round(x[2]),dtype=int)
    m =outlier_idxs[m]
    t = x[6]
    o =np.linspace(round(x[3]), round(x[4]), num=round(x[5]),dtype=int)
    o =inlier_idxs[o]

    arr = np.concatenate((m, o))

    arr = set(arr)
    arr = np.array(list(arr))
    arr = arr.astype(int)
    a = src[arr]
    b = dst[arr]
    if (len(TFORM) == 0):
        tform2,newoutliers,typ2 = returnNewTransform2(b,a,3)
        new_outlier_idxs = np.nonzero(newoutliers)[0]
        if tform2 != None:
            #m = random.randint(1,2)
            LocalSimilarity = checkSimilarityScore([tform2],[t],BoxList) #this only pick 3 transform for now
            return LocalSimilarity
        else:
            return 10000000
    else:
        tform2,newoutliers,typ2 = returnNewTransform2(b,a,3)
        new_outlier_idxs = np.nonzero(newoutliers)[0]
        if tform2 != None:
            #m = random.randint(1,2)
            TFORM.append(tform2)
            Filter.append(t)
            LocalSimilarity = checkSimilarityScore(TFORM,Filter,BoxList) #this only pick 3 transform for now
            TFORM.pop()
            Filter.pop()
            return LocalSimilarity
        else:
            return 10000000

#--- MAIN ---------------------------------------------------------------------+

class Particle:
    def __init__(self,x0,inlier_idxs,outlier_idxs,src,dst,BoxList,num_dimensions,TFORM,Filter):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual
        self.num_dimensions = num_dimensions
       # vmax = len(inlier_idxs)-1
        for i in range(0,num_dimensions-1):
            self.velocity_i.append(random.uniform(-100,100))
            #self.velocity_i.append(random.uniform(-vmax,vmax))
            self.position_i.append(x0[i])
        self.velocity_i.append(random.uniform(-1,1))
        self.position_i.append(x0[i+1])

        self.a = inlier_idxs
        self.b = outlier_idxs
        self.c = src
        self.d = dst
        self.e = BoxList
        self.f = TFORM
        self.g = Filter
    # evaluate current fitness
    def evaluate(self,costFunc):
        self.err_i=costFunc(self.position_i,self.a,self.b,self.c,self.d,self.e,self.f,self.g)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.5       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=2        # social constant

        for i in range(0,self.num_dimensions):
            r1=random.random()
            r2=random.random()

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        for i in range(0,self.num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i]=bounds[i][0]
                
class PSO():
    def __init__(self,costFunc,x0,bounds,num_particles,maxiter,inlier_idxs,outlier_idxs,src,dst,BoxList,TFORM,Filter):
       # global num_dimensions
        self.bounds = bounds
        self.maxiter = maxiter
        self.num_dimensions=len(x0)
        self.err_best_g=-1                   # best error for group
        self.pos_best_g=[]                   # best position for group
        self.num_particles=num_particles
        # establish the swarm
        self.swarm=[]
        self.func = costFunc
        for i in range(0,num_particles):
            self.swarm.append(Particle(x0,inlier_idxs,outlier_idxs,src,dst,BoxList,self.num_dimensions,TFORM,Filter))
    def optimize(self):
        # begin optimization loop
        i=0
        while i < self.maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            print("stepping")
            for j in range(0,self.num_particles):
                self.swarm[j].evaluate(self.func)

                # determine if current particle is the best (globally)
                if self.swarm[j].err_i < self.err_best_g or self.err_best_g == -1:
                    self.pos_best_g=list(self.swarm[j].position_i)
                    self.err_best_g=float(self.swarm[j].err_i)
                    print(self.err_best_g)
            # cycle through swarm and update velocities and position
            for j in range(0,self.num_particles):
                self.swarm[j].update_velocity(self.pos_best_g)
                self.swarm[j].update_position(self.bounds)
            i+=1

        # print final results
        print ('FINAL')
        print (self.pos_best_g)
        return self.pos_best_g,self.err_best_g
    
    
MIN_MATCH_COUNT = 10

FLANN_INDEX_KDTREE = 1

Accuracy = 0

##use 50 for demo
#trial_num= 0

for trial_num in range(len(tr)):
    cue_image = tr[trial_num][1]
    choice_l_image = tr[trial_num][4]
    choice_r_image = tr[trial_num][5]
    correct = tr[trial_num][6]
    img_dir = 'images/'

    choice1 = choice_l_image
    choice2 = choice_r_image

                # Read the main image
    img_rgb = cv2.imread(img_dir+cue_image)
                # Convert it to grayscale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
                # Read the template
    template1 = cv2.imread(img_dir + choice1,0)
    template2 = cv2.imread(img_dir + choice2,0)    

    BoxList = []

    contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    sd = ShapeDetector()
    tW, tH = template1.shape[::-1]
    tW2, tH2 = template2.shape[::-1]
    template2 = cv2.resize(template2, (tW, tH))
    ratio = 1

        # loop over the contours
    whiteBox = []
    bb = []
    for c in contours:
            # compute the center of the contour, then detect the name of the
            # shape using only the contour
        M = cv2.moments(c)
        shape = sd.detect(c)
            #print(shape)
        if (shape != "rectangle"):
            pass
        else:
            # multiply the contour (x, y)-coordinates by the resize ratio,
            # then draw the contours and the name of the shape on the image
            rect = cv2.boundingRect(c)
                #if ((rect[2] - tW) > -10) or ((rect[3] - tH) > -10): continue
                #print (cv2.contourArea(c))
                # show the output image
            x,y,w,h = cv2.boundingRect(c)

            if ((tW*tH)/(w*h) < 1.5):
                whiteBox.append(cv2.countNonZero(img_gray[y-10:y-10+tW,x-10:x-10+tH]))
                bb.append(rect)
            else:
                pass

    max_value = max(whiteBox)
        #white box detection
    max_index = whiteBox.index(max_value)

    x,y,w,h = bb[max_index]
    t1 = img_gray.copy()
    t1[y:y+tH,x:x+tW] = template1
    ##anomaly detection##
    Img_w = img_gray.shape[1]

    spacing = int(Img_w/4)
    #Width = int((Img_w - tW)/3.2)

    Width = spacing#int(tW+tW/3) #int((Img_w - tW)/3.2)

    x1 = 0
    x2 = spacing
    x3 = 2*spacing
    x4 = 3*spacing

    t4 = img_gray.copy()
    t3 = img_gray.copy()
    t4[:,x1:x1+Width] = 0

    box1 = t3[:,x1:x1+Width] 

    t4 = img_gray.copy()

    t4[:,x2:x2+Width] = 0

    box2 = t3[:,x2:x2+Width] 
    t4 = img_gray.copy()
    t4[:,x3:x3+Width] = 0
    box3 = t3[:,x3:x3+Width]


    BoxList.append(box1)
    BoxList.append(box2)
    BoxList.append(box3)
    BoxList.append(bb[max_index])

    #ADD THIS FOR FEATURE DETECTION
    box1[:,0:2] =0
    box1[0:2,:] = 0
    box1[:,-2:] = 0
    box1[-2:,:] = 0

    box2[:,0:2] = 0
    box2[0:2,:] = 0
    box2[:,-2:] = 0
    box2[-2:,:] = 0

    box3[:,0:2] = 0
    box3[0:2,:] = 0
    box3[:,-2:] = 0
    box3[-2:,:] = 0

    t4 = img_gray.copy()

    t4[:,x4:x4+Width] = 0
    box4 = t3[:,x4:x4+Width]

    ##STRATEGY 2
    good1 = []
    orb = cv2.ORB_create()
    ##########TRY TO get your own FEATURES INSTEAD!!!!!!!!!
    sift = cv2.xfeatures2d.SIFT_create()
    # Find keypoints and descriptors directly

    try:
        kp1, des1 = sift.detectAndCompute(box1,None) 
        kp2, des2 = sift.detectAndCompute(box2,None) 
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
        good1 = []
        for m,n in matches:
            if m.distance < 0.9*n.distance:
                good1.append(m)
    except:
        kp1, des1 = orb.detectAndCompute(box1,None) 
        kp2, des2 = orb.detectAndCompute(box2,None) 

        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
        nn_matches = matcher.knnMatch(des1, des2, 2)
        matched1 = []
        matched2 = []
        nn_match_ratio = 0.8 # Nearest neighbor matching ratio
        for m, n in nn_matches:
            if m.distance < nn_match_ratio * n.distance:
                good1.append(m)
    src = np.float32([ kp1[m.queryIdx].pt for m in good1 ]).reshape(-1,2)
    dst = np.float32([ kp2[m.trainIdx].pt for m in good1 ]).reshape(-1,2)

    #--- RUN ----------------------------------------------------------------------+

    tform1,outliers,typ = returnNewTransform1(dst,src)
    outlier_idxs = np.nonzero(outliers == 1)[0]
    inlier_idxs = np.nonzero(outliers == 0)[0]

    num_inliers = len(inlier_idxs)
    new_outlier_length = len(outlier_idxs)
    old_outlier_length = 0
    globalSimilarity = float('inf')
    old_local_sim = float('inf')
    import random

    if (len(outlier_idxs) > 2):
            #while similarity < 0.7:
        TFORM = []
        Filter = []
        sn = 3 #int(len(inlier_idxs)/3)
            #SHOULD THIS INITIAL CONDITION BE FIXED OR DYNAMICS?
            ###WHERE AND HOW MANY!!!!
        j = 0
        while (len(outlier_idxs) > 2): #if the actual list of outliers being updated reached below 2 stop, else
            j = j + 1
            if j > 2:
                break
                ##RANDOMLY SAMPLE INLIERs w PSO
            #np.linspace(0, 10, num=5)   
            initial = [0,len(outlier_idxs)-1,20,0,len(inlier_idxs)-1,2,0] # initial starting location [x1,x2...]
            bounds = [(0,len(outlier_idxs)-1),(0,len(outlier_idxs)-1),(2,len(outlier_idxs)-1),(0,len(inlier_idxs)-1),(0,len(inlier_idxs)-1),(2,len(inlier_idxs)-1),(0,1)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
            ##NEED TO ACCUMULATE !!!!!
            ###SO MSE REDUCE AND NOT INCREASE OVER TIME
            ###
            m = PSO(Function2Optimize,initial,bounds,100,30,inlier_idxs,outlier_idxs,src,dst,BoxList,TFORM,Filter)
            x, err=m.optimize()
            m=np.linspace(round(x[0]), round(x[1]), num=round(x[2]),dtype=int) 
            m =outlier_idxs[m]

            t = x[6]
            o=np.linspace(round(x[3]), round(x[4]), num=round(x[5]),dtype=int) 
            o =inlier_idxs[o]
            arr = np.concatenate((m, o))
            arr = set(arr)
            arr = np.array(list(arr))
            arr = arr.astype(int)
            a = src[arr]
            b = dst[arr]
            tform2,newoutliers,typ2 = returnNewTransform2(b,a,3)
            if tform2 != None:
                new_outlier_idxs = np.nonzero(newoutliers)[0]
                TFORM.append(tform2)
                #m = random.randint(1,2)
                Filter.append(t)
                #what if we dont try to replace outlier set w new outliers???
                #outlier_idxs = new_outlier_idxs    
                #print(len(outlier_idxs))        
        out = applyRule(TFORM,Filter,BoxList[0])    
        out1 = applyRule(TFORM,Filter,BoxList[2])    
        out2 = applyRule(TFORM,Filter,out1)  
        out3 = applyRule(TFORM,Filter,out2)  
    else:
        #instead of warping input1...
        out = (warp(BoxList[0], tform1,cval=1, mode ='constant'))
                #instead of warping input1...
        out1 = warp(BoxList[2], tform1,cval=1, mode ='constant')
                #instead of warping input1...
        out2 = warp(out1, tform1,cval=1, mode ='constant')
                #instead of warping input1...
        out3 = warp(out2, tform1,cval=2, mode ='constant')

    template1 = np.array(template1, dtype = np.uint8)
    template2 = np.array(template2, dtype = np.uint8)
    t3 = img_gray.copy()
    x4 = spacing*3
    t3[:,x4:x4+Width] = out1
    result = t3[y:y+tH,x:x+tW]
    Answer1 = mse(result,template1)
    Answer2 = mse(result,template2)

    if (Answer1 < Answer2): #if greater sim than mse
        choice = 0
    else:
        choice = 1
    print("Task " + str(trial_num))
    if (int(choice) == int(correct)):
        print("MSE Correct")
    else:
        print("MSE Wrong")

