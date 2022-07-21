"""
功能描述：
1.传入特定时刻的env，渲染出RGB图，可以选择，是否将其保存为一个小视频
2.需要用pygame可视化当前图
3.不需要pygame乱七八糟的功能
4.视频保存路径和当前实验log路径一致
5.视频名称需要标注好epoch

"""


import pygame
import os
from pygame.locals import *
from sys import exit
import numpy as np
import cv2
import imutils


class GymRenderImageSaveVideoClass:
    def __init__(self,
                 exp_path='',
                 fps=20,
                 args=None,
                 ):
        self.args = args
        self.exp_path = exp_path
        try:
            os.mkdir(self.exp_path)
        except Exception as e:
            print(e)
        self.start_epoch = 0
        self.max_store = 49
        # save video:
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
        self.fps = fps

        self.image_width = 256
        self.image_height = 256
        self.start_epoch = 0
        # pygame init
        pygame.init()
        self.break_flag = False
        self.image_count = 0
        self.position_font = pygame.font.SysFont("幼圆", 40)
        # 返回一个窗口Surface对象
        self.screen = pygame.display.set_mode((self.image_width,
                                               self.image_height),
                                              0, 32)
        self.exp_name = 'name'

    def update(self, image, epoch=0, step=0, reward=0):
        # 判断是否退出
        for event in pygame.event.get():
            if event.type == QUIT:
                self.break_flag = True
        # 在窗口标题上显示参数元组
        pygame.display.set_caption(self.exp_name)
        robot_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # 图片可能需要旋转，不需要的话就注释掉
        robot_image = imutils.rotate(robot_image, 90, )
        robot_image = cv2.resize(robot_image,
                                 (self.image_height,
                                  self.image_width))

        my_surface = pygame.pixelcopy.make_surface(robot_image)
        self.screen.blit(my_surface, (0, 0))
        pygame_str = 'epoch:' + str(epoch) + '-step:' + str(step) + '-rew:' + str(np.round(reward, 3))
        text1 = self.position_font.render(pygame_str, True,
                                          (255, 0, 0), (0, 0, 0))
        self.screen.blit(text1,
                         [10, 40])

        pygame.display.update()
        return robot_image

    def step(self, epoch='', step='', reward=0.0,
             env=None, specfic='', per_epoch=False,
             ):
        if self.image_count == 0:
            video_name = self.args.exp_name + '_st_ep' + str(int(epoch))
            video_path = os.path.join(self.exp_path, specfic+'_'+video_name + '.avi')
            self.out = cv2.VideoWriter(video_path, self.fourcc, self.fps,
                                       (self.image_width,
                                        self.image_height))
        if env is None:
            robot_image = np.uint8(np.random.random((1200, 1080, 3)))
        else:
            env.render("human")
            robot_image = env.render("rgb_array",
                                     width=1200,
                                     height=1080,
                                     )
        # 判断是否退出
        for event in pygame.event.get():
            if event.type == QUIT:
                self.break_flag = True
        if self.break_flag:
            pygame.quit()
            exit()
        robot_image = robot_image[150:-100, 300:, :]
        robot_image = cv2.cvtColor(robot_image, cv2.COLOR_RGB2BGR)

        self.update(image=robot_image,
                    epoch=epoch,
                    step=step,
                    reward=reward
                    )
        post_img = cv2.resize(robot_image,
                              (self.image_width,
                               self.image_height))
        fontScale = 0.6
        text_thickness = 2
        bg_color = (255, 0, 0)
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(post_img, 'epoch:' + str(epoch),
                    (10, 20), fontFace, fontScale,
                    bg_color, text_thickness, 8)
        cv2.putText(post_img, 'step:' + str(step),
                    (10, 40), fontFace, fontScale,
                    bg_color, text_thickness, 8)
        cv2.putText(post_img, 'rew:' + str(np.round(reward, 3)),
                    (10, 60), fontFace, fontScale,
                    bg_color, text_thickness, 8)
        post_img = np.uint8(post_img)

        self.out.write(post_img)
        self.image_count += 1
        if per_epoch:
            if step == self.max_store - 1:
                self.image_count = 0
                self.out.release()


def main():
    import time
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--max_steps', type=int, default=20)
    parser.add_argument('--transfer_epoch', type=int, default=500)
    parser.add_argument('--exp_name', type=str, default='TD3_lift_restore')
    parser.add_argument('--output_dir', type=str, default='Real_Lift_State')

    args = parser.parse_args()
    logger_kwargs = {'exp_name': 'TD3_lift_restore',
                     'output_dir': 'Real_Lift_State'}
    dpc = GymRenderImageSaveVideoClass(exp_path=logger_kwargs['output_dir'],
                                       args=args,
                                       )
    test_dpc = GymRenderImageSaveVideoClass(exp_path=logger_kwargs['output_dir'],
                                            args=args,)
    import matplotlib.pyplot as plt
    st_list = []
    import gym
    env = gym.make("FetchPush-v1",
                   reward_type='dense'
                   )

    for i in range(args.epochs):
        env.reset()
        for j in range(args.max_steps):
            st = time.time()
            obs, r, done, info = env.step(env.action_space.sample())
            dpc.step(epoch=i, step=j, reward=r,
                     env=env, specfic='test',
                     per_epoch=False)
            test_dpc.step(epoch=i, step=j, reward=r,
                          env=env, specfic='train',
                          per_epoch=False)
            print("step:", time.time() - st)
            st_list.append(time.time() - st)
            if i % 10 == 0 and j % 100 == 0:
                plt.plot(st_list)
                plt.pause(1)
            if dpc.break_flag:
                pygame.quit()
                dpc.out.release()
                cv2.destroyAllWindows()
                exit()
    dpc.out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


