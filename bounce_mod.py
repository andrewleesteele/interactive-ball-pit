import cv2
import cv
import numpy as np
import math

import time

class Ball:
    def __init__(self, pos, rad, vel):
        self.pos = pos
        self.vel = vel
        self.rad = rad
        self.ipos = None
        self.doffs = None
        self.doffs_prev = None
        self.wrad = int(1.5*self.rad+20)
        self.wsz = 2*self.wrad + 1
        self.subimage = 255*np.ones((self.wsz,self.wsz),dtype='uint8')
        self.dists = None
        self.dists_prev = None
        self.dists_smoothed = None
        self.color = ((int)(255*np.random.rand()),(int)(255*np.random.rand()),(int)(255*np.random.rand()))

########################################################################
# This is a accessor function that returns the color of the ball
########################################################################
    def getColor(self):
        return self.color

########################################################################
# This is a accessor function that returns the position of the ball
########################################################################
    def getPos(self):
        return self.pos

########################################################################
# This is a modifier function that sets the position of the ball
########################################################################
    def setPos(self, new_pos):
        self.pos = new_pos

########################################################################
# This is a accessor function that returns the velocity of the ball
########################################################################
    def getVel(self):
        return self.vel

########################################################################
# This is a modifier function that sets the velocity of the ball
########################################################################
    def setVel(self, new_vel):
        self.vel = new_vel

########################################################################
# This is a accessor function that returns the radius of the ball
########################################################################
    def getRad(self):
        return self.rad

########################################################################
# This is a modifier function that sets the radius of the ball
########################################################################
    def setRad(self, radius):
        self.rad = rad

########################################################################
# This is a accessor function that returns the degrees of freedom of the ball
########################################################################
    def getDoffs(self):
        return self.doffs

########################################################################
# This is a modifier function that sets the degrees of freedom of the ball
########################################################################
    def setDoffs(self, doffs):
        self.doffs = doffs

########################################################################
# This is a accessor function that returns the previous degrees of freedom of the ball
########################################################################
    def getDoffsPrev(self):
        return self.doffs_prev

########################################################################
# This is a modifier function that sets the previous degrees of freedom of the ball
########################################################################
    def setDoffsPrev(self, new_doffs_prev):
        self.doffs_prev = new_doffs_prev

########################################################################
# This is a accessor function that returns the radius of the sub window
########################################################################
    def getWrad(self):
        return self.wrad

########################################################################
# This is a accessor function that returns the sub image
########################################################################
    def getSubIm(self):
        return self.subimage

########################################################################
# This is a modifier function that sets the sub image
########################################################################
    def setSubIm(self, new_sub):
        self.subimage = new_sub

########################################################################
# This is a accessor function that returns the distance of the ball from obstacles
########################################################################
    def getDists(self):
        return self.dists

########################################################################
# This is a modifier function that sets the distance of the ball from obstacles
########################################################################
    def setDists(self, new_dist):
        self.dists = new_dist

########################################################################
# This is a accessor function that returns smoothed distance of the ball from obstacles
########################################################################
    def getDistsSmoothed(self):
        return self.dists_smoothed

########################################################################
# This is a modifier function that sets the smoothed distance of the ball from obstacles
########################################################################
    def setDistsSmoothed(self, new_smooth):
        self.dists_smoothed = new_smooth

########################################################################
# This is a accessor function that returns the previous distances
########################################################################
    def getDistsPrev(self):
        return self.dists_prev

########################################################################
# This is a modifier function that sets the previous distances
########################################################################
    def setDistsPrev(self, new_dist_prev):
        self.dists_prev = new_dist_prev

########################################################################
# This is a function that extracts the sub image and returns it
########################################################################
    def extractSubimage(self, img, ctr, default=255):

        #window size
        wsz = 2*self.wrad + 1

        h,w = img.shape[:2]

        ix,iy = ctr

        ix0 = ix-self.wrad
        ix1 = ix0+wsz
        iy0 = iy-self.wrad
        iy1 = iy0+wsz

        #make sure the sub image is within img boundaries
        ax0 = max(0, min(ix0, w))
        ax1 = max(0, min(ix1, w))
        ay0 = max(0, min(iy0, h))
        ay1 = max(0, min(iy1, h))

        #find the bounds of the sub image
        sy0 = ay0-iy0
        sy1 = sy0 + ay1-ay0
        sx0 = ax0-ix0
        sx1 = sx0 + ax1-ax0

        self.subimage[:] = default
        self.subimage[sy0:sy1, sx0:sx1] = img[ay0:ay1, ax0:ax1]


class BallSim:

    def __init__(self, new_scene, num_balls):

        self.window = 'bounce'

        #invert image
        self.scene = 255-new_scene

        #convert greyscale to RGB
        self.scene_rgb = cv2.cvtColor(self.scene, cv2.COLOR_GRAY2RGB)
        self.h, self.w = self.scene.shape[:2]
        self.ball_radius = 100

        self.all_balls = []

        #create the amount of balls specified with a random location and velocity
        for i in range(num_balls):
            pos = np.array([np.random.rand()*self.w, np.random.rand()*self.h], dtype='f')
            rad = self.ball_radius
            vel = (np.random.rand(2)*2-1)*1000
            new_ball = Ball(pos, rad, vel)
            self.all_balls.append(new_ball)


        #physics variables
        self.gravity = np.array([0,400])
        self.k_restitution = 0.8
        self.k_friction = 0.99
        self.k_collision = 0.2
        self.k_rvel = 1

        self.draw_shift = 2
        self.draw_scale = 1<<self.draw_shift

        self.kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='f')/8
        self.ky = self.kx.transpose()

        #frame display variables
        self.dt_msec = 16
        self.delay_msec = 4
        self.updates_per_frame = 4
        self.frame_dt = self.dt_msec * 1e-3
        self.update_dt = self.frame_dt / self.updates_per_frame

        self.sample_subpix = None

        self.display = np.empty_like(self.scene_rgb)
        self.overlay = 255*np.ones_like(self.scene)
        self.combined = np.empty_like(self.scene)

########################################################################
# This is a function updates the background image
########################################################################
    def updateScene(self, new_scene):

        #loop through each ball
        for i in range(len(self.all_balls)):
            #get position of ball
            pos_int = tuple(np.round(self.draw_scale*self.all_balls[i].getPos()).astype(int))
            #get radius of ball
            rad_int = np.round(self.draw_scale*self.ball_radius)    

            #draw black circle onto scene in order to black out circles identified as ecluded objects
            cv2.circle(new_scene, pos_int, rad_int,
                       0, -1, cv2.CV_AA, self.draw_shift)

        #remove noise from image
        kernel = np.ones((10,10),np.uint8)
        new_scene = cv2.erode(new_scene,kernel,iterations=1)
        kernel = np.ones((20,20),np.uint8)
        new_scene = cv2.dilate(new_scene,kernel,iterations=1)

        #invert image
        self.scene = 255-new_scene

        #add a black border to image
        cv2.rectangle(self.scene,(0,0),(self.w, self.h),0,30)
        
        #convert greyscale to RGB
        self.scene_rgb = cv2.cvtColor(self.scene, cv2.COLOR_GRAY2RGB)


########################################################################
# This is a function that checks a ball's subimage for potential obstacles
########################################################################
    def sample(self, d, pos):
        if ( pos[0] < 0 or pos[0] >= d.shape[1] or
             pos[1] < 0 or pos[1] >= d.shape[0] ):
            return 0.0, 0.0, 0.0
        self.sample_subpix = cv2.getRectSubPix(d, (3,3), tuple(pos), 
                                               self.sample_subpix)

        dist_to_nearest = self.sample_subpix[1,1]
        gx = (self.sample_subpix*self.kx).sum()
        gy = (self.sample_subpix*self.ky).sum()
        return dist_to_nearest, gx, gy
    

########################################################################
# This is a function that normalizes v
########################################################################
    def normalize(self, v):
        vn = np.linalg.norm(v)
        if vn:
            v /= vn
        return v

########################################################################
# This is a function runs the physics simulation of the balls
########################################################################
    def run(self):
        
        for ts in range(1):
            t = time.time()

            #perform  a bitwise and operation on scene and overlay in order to combine images
            np.bitwise_and(self.scene, self.overlay, self.combined)

            ipos = None
            for i in range(len(self.all_balls)):

                #position and radius of ball
                ipos = tuple(np.round(self.all_balls[i].getPos()).astype(int))
                wrad = self.all_balls[i].getWrad()

                #set degrees of freedom
                self.all_balls[i].setDoffs(np.array([wrad, wrad],dtype='f')-ipos)
                
                #extract subimage of ball
                self.all_balls[i].extractSubimage(self.combined, ipos)

                #set distance of potential objects in sub image
                self.all_balls[i].setDists(cv2.distanceTransform(self.all_balls[i].getSubIm(), cv.CV_DIST_L2, 
                                    cv.CV_DIST_MASK_PRECISE, self.all_balls[i].getDists()))
                
                #set smoothed distance of potential objects in sub image
                self.all_balls[i].setDistsSmoothed(cv2.GaussianBlur(self.all_balls[i].getDists(), (0,0), 0.5, 
                    self.all_balls[i].getDistsSmoothed()))

                #if prev distances hasn't been set, copy current distances as previous distances
                if self.all_balls[i].getDistsPrev() is None:
                    self.all_balls[i].setDistsPrev(self.all_balls[i].getDists().copy())
                    self.all_balls[i].setDoffsPrev(self.all_balls[i].getDoffs().copy())
            
                for j in range(self.updates_per_frame):
                    #apply physics engine to ball
                    vnew = self.all_balls[i].getVel() + self.update_dt*self.gravity
                    mod = 0.5*self.update_dt*(self.all_balls[i].getVel() + vnew)
                    self.all_balls[i].setPos(self.all_balls[i].getPos() + mod)
                    self.all_balls[i].setVel(vnew)

                    #find nearest distance to obstacle
                    dist_to_nearest, gx, gy = self.sample(self.all_balls[i].getDistsSmoothed(), 
                                                          self.all_balls[i].getPos()+self.all_balls[i].getDoffs())
                    #if ball is touching an obstacle
                    if dist_to_nearest < self.ball_radius:
                        normal = self.normalize(np.array(( gx, gy )))
                        
                        dprev, gpx, gpy = self.sample(self.all_balls[i].getDistsPrev(), 
                                        self.all_balls[i].getPos()+self.all_balls[i].getDoffsPrev())
                        
                        nprev = self.normalize(np.array( (gpx, gpy) ))
                        fvel = -self.k_rvel * (dist_to_nearest*normal - dprev * nprev) / self.frame_dt


                        rvel = self.all_balls[i].getVel() - fvel

                        proj = np.dot(normal, rvel)
                        mod = self.all_balls[i].getPos() + self.k_collision*dist_to_nearest*normal
                        self.all_balls[i].setPos(mod)
                        if proj < 0:
                            rvel_normal = normal * proj
                            rvel_tangent = rvel - rvel_normal
                            rvel = self.k_friction*rvel_tangent - self.k_restitution*rvel_normal
                            self.all_balls[i].setVel(rvel + fvel)

                    #reset ball's position
                    self.all_balls[i].setPos(np.maximum((0,0), np.minimum(self.all_balls[i].getPos(), (self.w, self.h))))

            cv2.cvtColor(self.combined, cv2.COLOR_GRAY2RGB, self.display)

            img = 0*np.ones_like(self.display)

            for i in range(len(self.all_balls)):
                pos_int = tuple(np.round(self.draw_scale*self.all_balls[i].getPos()).astype(int))
                rad_int = np.round(self.draw_scale*self.ball_radius)    

                #draw circle onto image with all the balls
                cv2.circle(img, pos_int, rad_int,
                           self.all_balls[i].getColor(), -1, cv2.CV_AA, self.draw_shift)

                self.all_balls[i].setDistsPrev(self.all_balls[i].getDistsSmoothed())
                self.all_balls[i].setDoffsPrev(self.all_balls[i].getDoffs())

            #display the image and move to top corner
            cv2.imshow(self.window, img)
            cv2.moveWindow(self.window,0,0)
            
            k = cv2.waitKey(self.dt_msec - self.delay_msec)
