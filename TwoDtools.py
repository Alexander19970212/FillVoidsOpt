import numpy as np
import matplotlib.pyplot as plt

import matplotlib.patches as patches
from scipy.interpolate import interp2d
from scipy.optimize import minimize

from scipy import integrate
import time 

import math, random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

from IPython.display import clear_output
import matplotlib.pyplot as plt

def get_init_x(map, stride = 4, window_size = 4, drowing=False):

    conv = np.zeros((33, 33))
    
    for i in range(33):
        for j in range(33):
            i_im = i + stride
            j_im = j + stride
            conv[i, j] = map[i_im - stride: i_im + stride, j_im - stride: j_im + stride].sum()
    
    
    min_ind = np.unravel_index(conv.argmin(), conv.shape)
    # print(min_ind)

    if drowing:
        plt.scatter(min_ind[1], min_ind[0])
        plt.imshow(conv)
    
    x_center = min_ind[0] + stride
    y_center = min_ind[1] + stride
    
    square_coords = np.array([x_center - window_size, x_center + window_size, y_center - window_size, y_center + window_size])
    square_coords = np.clip(square_coords, 0, 40)
    
    x_init = np.array([square_coords[0], square_coords[1] - square_coords[0], square_coords[2], square_coords[3] - square_coords[2]])

    return x_init, square_coords

def integral_domain_triangular(coords):
    # sort points
    x_coords = coords[:, 0]
    sort_indeces = np.argsort(x_coords)
    # print(sort_indeces)
    coords = coords.copy()[sort_indeces]
    left_point = coords[0]

    k_2 = (coords[2, 1] - coords[0, 1])/(coords[2, 0] - coords[0, 0])
    b_2 = coords[0, 1] - k_2 * coords[0, 0]

    if left_point[0] != coords[1, 0]:
        # find tangent cooefs
        k_1 = (coords[1, 1] - coords[0, 1])/(coords[1, 0] - coords[0, 0])
        b_1 = coords[0, 1] - k_1 * coords[0, 0]
        
        # find lower_line
        if k_1 < k_2:
            lower_point = coords[1]
            upper_point = coords[2]
    
        else:
            lower_point = coords[2]
            upper_point = coords[1]

    else:
        if coords[1, 1] < coords[2, 1]:
            lower_point = coords[1]
            upper_point = coords[2]
    
        else:
            lower_point = coords[2]
            upper_point = coords[1]
            

    if coords[1, 0] != coords[2, 0]:
        k_3 = (lower_point[1] - upper_point[1])/(lower_point[0] - upper_point[0])
        b_3 = upper_point[1] - k_3 * upper_point[0]
        
    bounds = []

    if left_point[0] != coords[1, 0]:
        if (coords[1] == lower_point).all():
            case = [[left_point[0], coords[1,0]], [k_1, b_1], [k_2, b_2]]
        else:
            case = [[left_point[0], coords[1,0]], [k_2, b_2], [k_1, b_1]]

        bounds.append(case)

    if coords[1, 0] != coords[2, 0]:

        if k_3 > k_2:
            case = [[coords[1, 0], coords[2,0]], [k_3, b_3], [k_2, b_2]]
        else:
            case = [[coords[1, 0], coords[2,0]], [k_2, b_2], [k_3, b_3]]

        bounds.append(case)

    return np.array(bounds)

def f_simp_area(x, y):
    return 1

def f_area(first_triag, second_triag):
    # first_triag
    domains = integral_domain_triangular(first_triag)
    s_1 = 0
    for domain in domains:
        s_1 += integrate.dblquad(f_simp_area, 
                                 domain[0, 0], 
                                 domain[0, 1], 
                                 lambda x: x * domain[1, 0] + domain[1, 1], 
                                 lambda x: x * domain[2, 0] + domain[2, 1],
                                 epsabs=1.49e-03)[0]

    # first_triag
    domains = integral_domain_triangular(second_triag)
    s_2 = 0
    for domain in domains:
        s_2 += integrate.dblquad(f_simp_area, 
                                 domain[0, 0], 
                                 domain[0, 1], 
                                 lambda x: x * domain[1, 0] + domain[1, 1], 
                                 lambda x: x * domain[2, 0] + domain[2, 1],
                                 epsabs=1.49e-03)[0]

    return s_1 + s_2

def fill_triangulars(first_triag, second_triag):
    # first_triag
    domains_1 = integral_domain_triangular(first_triag)

    m = 40
    all_coords = np.array([[i, j] for i in range(m) for j in range(m)])
    all_coords = all_coords + 0.5
    

    ####################################
    
    # all_coords_1 = all_coords.copy()
    coords_cases_1 = []
    # print(domains)
    for domain in domains_1:
        # print("domains 1", domain)
        all_coords_1 = all_coords.copy()
        all_coords_1 = all_coords_1[all_coords_1[:,0] >= domain[0, 0]]
        all_coords_1 = all_coords_1[all_coords_1[:, 0] <= domain[0, 1]]

        all_coords_1 = all_coords_1[all_coords_1[:, 0] * domain[1, 0] + domain[1, 1] <= all_coords_1[:, 1]]
        all_coords_1 = all_coords_1[all_coords_1[:, 0] * domain[2, 0] + domain[2, 1] >= all_coords_1[:, 1]]

        coords_cases_1.append(all_coords_1)

    # all_coords_1 = np.concatenate((coords_cases_1[0], coords_cases_1[1]), axis=0)
    
    # first_triag
    domains_2 = integral_domain_triangular(second_triag)
    
    # all_coords_2 = all_coords.copy()
    coords_cases_2 = []
    for domain in domains_2:
        all_coords_2 = all_coords.copy()
        # print("domains 2", domain)
        all_coords_2 = all_coords_2[all_coords_2[:,0] > domain[0, 0]]
        # print("cutting 1", all_coords_2.shape)
        all_coords_2 = all_coords_2[all_coords_2[:, 0] < domain[0, 1]]
        # print("cutting 2", all_coords_2.shape)
        all_coords_2 = all_coords_2[all_coords_2[:, 0] * domain[1, 0] + domain[1, 1] <= all_coords_2[:, 1]]
        # print("cutting 3", all_coords_2.shape)
        all_coords_2 = all_coords_2[all_coords_2[:, 0] * domain[2, 0] + domain[2, 1] >= all_coords_2[:, 1]]
        # print("cutting 4", all_coords_2.shape)
        coords_cases_2.append(all_coords_2)

    # all_coords_2 = np.concatenate((coords_cases_2[0], coords_cases_2[1]), axis=0)
    # try:
    #     filling_coords = coords_cases_1[0]
    # except:
    #     print(domains_1)
    #     print(first_triag)
    #     filling_coords = coords_cases_1[0]
    
    for i, coords_case in enumerate(coords_cases_1):
        if i == 0:
            filling_coords = coords_cases_1[0]
        else:
            filling_coords = np.concatenate((filling_coords, coords_case), axis=0)

    for i, coords_case in enumerate(coords_cases_2):
        if len(coords_cases_1) == 0 and i == 0:
            filling_coords = coords_cases_2[0]
        else:    
            filling_coords = np.concatenate((filling_coords, coords_case), axis=0)
            
            
    
    # print("All coords 1: ", all_coords_1.shape)
    # print("All coords 2: ", all_coords_2.shape)
    # filling_coords = np.concatenate((all_coords_1, all_coords_2), axis=0)
    # print(filling_coords)

    try:
        filling_coords[:, :] -= 0.5
    
        # print("All coords: ", filling_coords.shape)
        fiiling_coords = filling_coords.astype(int)
    
        rows = np.array(filling_coords[:, 0], dtype=np.intp)
        colomns = np.array(filling_coords[:, 1], dtype=np.intp)
    
        # print(rows)
    
        # result_matrix = np.zeros((m, m))
        # result_matrix[rows, colomns] = 1
    
        return rows, colomns
    except:
        print("Empty case")
        return [], []

def fill_2d_with_polygon4(args):
    # first way
    first_triangular_1 = np.array([[args[0], args[1]], [args[2], args[3]], [args[4], args[5]]])
    second_triangular_1 = np.array([[args[0], args[1]], [args[4], args[5]], [args[6], args[7]]])
    area_1 = f_area(first_triangular_1, second_triangular_1)

    # second way
    first_triangular_2 = np.array([[args[0], args[1]], [args[2], args[3]], [args[6], args[7]]])
    second_triangular_2 = np.array([[args[2], args[3]], [args[4], args[5]], [args[6], args[7]]])
    area_2 = f_area(first_triangular_2, second_triangular_2)
    
    area = min(area_1, area_2)

    if area == area_1:
        # print("first_case")
        first_triangular = first_triangular_1
        second_triangular = second_triangular_1
    
    else:
        # print("second_case")
        first_triangular = first_triangular_2
        second_triangular = second_triangular_2

    # importance_map_test = importance_map.copy()
    # try:
    rows, colomns = fill_triangulars(first_triangular, second_triangular)
    # except:
    #     print(first_triangular, second_triangular)
    #     rows, colomns = fill_triangulars(first_triangular, second_triangular)
    # importance_map_test[rows, colomns] = 1
    # plt.imshow(importance_map_test)
    # plt.show()
    return rows, colomns

def pol2x(coords):

    x = [coords[0,0], coords[0,1], 
         coords[1,0], coords[1,1], 
         coords[2,0], coords[2,1], 
         coords[3,0], coords[3,1]]

    return x
