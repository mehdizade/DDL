from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os.path
import os
import numpy as np
from tensorflow.python.framework import ops
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#_module=tf.load_op_library(tf.sysconfig.get_lib()+'/python/user_ops/lasso_fista.so')
#_zero_out = _zero_out_module.zero_out




#def lasso_fista(Dict,input,lambda1,ksizes,strides,rates,padding,maxItr=50,NonNeg=True, debug=False, name='lasso'):
def lasso_fista_col(X,Dict,lambda1,maxItr,NonNeg,debug):
   num_patch= tf.shape(X)[1]
   num_atoms=tf.shape(Dict)[1]

   DtD=tf.matmul(a=Dict,b=Dict,transpose_a=True,transpose_b=False)
   DtX=tf.matmul(a=Dict,b=X,transpose_a=True,transpose_b=False)


   self_adj=tf.self_adjoint_eigvals(DtD)
      #L= tf.reduce_max(self_adj)
   L=self_adj[-1]
   Linv = 1/L
   lambdaLinv = lambda1*Linv

   #print([num_atoms,num_patch])
   #num_patch= tf.shape(X)[1]

   A_old=tf.zeros(tf.stack([num_atoms,num_patch]),X.dtype)
   X_old=A_old+0
   t_old=1
      #t_old=tf.constant(1,dtype=tf.float32)

      #print(type(num_atoms))
   F= tf.eye(num_atoms,dtype=X.dtype)
   F=F-(Linv*DtD)
      #F= tf.subtract(x=F,y=Linv*DtD)
       
   const_A = tf.scalar_mul(scalar=Linv, x=DtX)
   const_A=const_A - lambdaLinv

   E_list=[]
   sp_list=[]
   for itr in range(maxItr):
      A_new = tf.matmul(a=F, b=X_old)
      A_new = tf.add(A_new,const_A)
      A_new = tf.multiply(tf.sign(A_new), tf.maximum(tf.abs(A_new)-lambdaLinv,0))
      if(NonNeg):
         A_new = tf.nn.relu(A_new)
            
      #else:
      
 
      t_new = 0.5 *(1+ math.sqrt(1 + 4*t_old**2))
      X_new = (1 + (t_old - 1)/t_new) * A_new  -  ((t_old - 1)/t_new) *A_old

      A_old = A_new+0
      t_old = t_new+0
      X_old = X_new+0

      if debug:
         tmp=tf.subtract(X,tf.matmul(Dict,A_new))
         E_list.append(tf.norm(tmp))
         sp_list.append( tf.norm(A_new,1) )

        #A_old=tf.transpose(A_old)
        
   return(A_old)

def lasso_fista(Dict,input,lambda1,ksizes,strides,rates,padding,maxItr=50,NonNeg=True, debug=False,name=None ):
   images_p=tf.extract_image_patches(images=input, ksizes=ksizes, strides=strides, rates=rates, padding=padding)
   #print(tf.shape(input))
   #(num_img,W_p,H_p,p_size)=tf.shape(image_p)
   num_img=tf.shape(images_p)[0]
   W_p=tf.shape(images_p)[1]
   H_p=tf.shape(images_p)[2]
   p_size=tf.shape(images_p)[3]
   num_atoms=tf.shape(Dict)[1]

   X=tf.reshape(images_p,[-1,p_size])
   X=tf.transpose(X)
        
   A_old=lasso_fista_col(X,Dict,lambda1,maxItr,NonNeg,debug)
   A_old=tf.transpose(A_old)
      
   A_old=tf.reshape(A_old,[num_img,W_p,H_p,num_atoms])
   tmp=tf.shape(input)
   return(A_old,tmp) 


####################################################################################
####################################################################################
#                                  Gradient                            
####################################################################################
def compute_lasso_grad(X, A, D, lambda1, grad_output, ksizes, strides, rates, padding, params, NonNeg):
   bsize=params[0] #10
   target_h=params[1] #16
   target_w=params[2] #16
   target_dep=params[3] #3
    
   [M,N]=D.shape
   #X=tf.transpose(X,[0,2,1,3])
   X_col=tf.extract_image_patches(images=X, ksizes=ksizes, strides=strides, rates=rates, padding=padding)
    
   X_col=tf.reshape(X_col, [-1, M])
   X_col=tf.transpose(X_col)
    
   #A=tf.transpose(A,[0,2,1,3])
   A_col=tf.reshape(A, [-1, N])
   A_col=tf.transpose(A_col)
  
   #grad_output=tf.transpose(grad_output,[0,2,1,3])
   grad_output_col=tf.reshape(grad_output, [-1, N])
   grad_output_col=tf.transpose(grad_output_col)
    
   [lambda1_grad, X_grad_col, D_grad]=compute_lasso_grad_col2(X_col, A_col, D, lambda1, grad_output_col, NonNeg)
     
   X_grad= col_to_image(X_grad_col,ksizes,strides,rates,padding, target_h=target_h,
                        target_w=target_h, target_dep=target_dep, bsize=bsize)
   #nrm = col_to_image(tf.ones_like(X_grad_col),ksizes,strides,rates,padding, target_h=target_h,
                   #    target_w=target_w, target_dep=target_dep, bsize=bsize)
   #X_grad=tf.divide(X_grad,nrm)
    
   return (lambda1_grad, X_grad, D_grad)


def compute_lasso_grad_col(X, A, D, lambda1, grad_output, NonNeg):
   sample_count=tf.shape(A)[1]
   num_atoms=tf.shape(D)[1]
    
   #finding the active set  
   if NonNeg:
      zero = tf.constant(0, dtype=X.dtype)
      #where = tf.greater_equal(A, zero)
      where = tf.greater(A, zero)
   else:
      zero = tf.constant(0, dtype=X.dtype)
      where = tf.not_equal(A, zero)
   max_act_col=tf.reduce_sum(tf.cast( where, tf.int32), 0)
   max_act=tf.reduce_max(max_act_col)
    
   def pinv(A, b, reltol=10**(-20)):
      # Compute the SVD of the input matrix A
      s, u, v = tf.svd(A)

      # Invert s, clear entries lower than reltol*s[0].
      atol = tf.reduce_max(s) * reltol
      s = tf.boolean_mask(s, s > atol)
        
      s_inv = tf.diag(tf.concat([1. / s, tf.zeros([tf.size(b) - tf.size(s)],dtype=A.dtype)], 0))
      out=tf.matmul(v, tf.matmul(s_inv, tf.matmul(u, tf.reshape(b, [-1, 1]), transpose_a=True)))     

      return out

   def cond(where, D, grad_output, output, i):
      return tf.less(i, tf.shape(where)[1])

   def body(where, D, grad_output, output, i):
      indices= tf.where(where[:,i])
      D_act = tf.transpose(tf.gather_nd(tf.transpose(D), indices))
               
      DTD_act=tf.matmul(D_act,D_act, transpose_a=True) + (10**(-6) * tf.eye(tf.shape(D_act)[1],dtype=X.dtype))
        
      grad_output_act=tf.gather_nd(grad_output[:,i], indices)
      B_act=pinv(DTD_act, grad_output_act)
      shape=tf.expand_dims(tf.shape(D)[1], 0)
      shape=tf.cast(shape, tf.int64)
      B = tf.SparseTensor(indices, tf.squeeze(B_act), shape)
      B=tf.sparse_tensor_to_dense(B)
      output = output.write(i, B)
        
      return where, D, grad_output, output, i + 1

   output_ta = tf.TensorArray(dtype=X.dtype, size=0, dynamic_size=True) 
   _, _, _, output_op, _  = tf.while_loop(cond, body, [where, D, grad_output, output_ta, 0])
    
   Beta = output_op.stack()
   Beta = tf.transpose(Beta,[1,0])
    
   lambda1_grad=-1*tf.reduce_sum(Beta) 
   X_grad_col=tf.matmul(D,Beta)  
    
   D_grad_tmp1=-1*tf.matmul(tf.matmul(D,Beta), A, transpose_b=True)
    
   E= X- tf.matmul(D,A)
   D_grad_tmp2=tf.matmul(E, Beta, transpose_b=True)
   D_grad=D_grad_tmp1+D_grad_tmp2
    
    
   return(lambda1_grad, X_grad_col, D_grad)


def compute_lasso_grad_col2(X, A, D, lambda1, grad_output, NonNeg):
    
   sample_count=tf.shape(A)[1]
   num_atoms=tf.shape(D)[1]
    
   #finding the active set  
   if NonNeg:
      zero = tf.constant(0, dtype=X.dtype)
      #where = tf.greater_equal(A, zero)
      where = tf.greater(A, zero)
   else:
      zero = tf.constant(0, dtype=X.dtype)
      where = tf.not_equal(A, zero)
    
   max_act_col=tf.reduce_sum(tf.cast(where, tf.int32),0)
   max_act=tf.reduce_max(max_act_col,keep_dims=True)
   min_act=tf.reduce_min(max_act_col,keep_dims=True)
    
   pad_count=max_act-min_act
   pad_count=tf.squeeze(pad_count)
    
   stich_bank=tf.matrix_band_part(tf.ones([pad_count,pad_count], tf.int32), 0, -1)
   stich_bank=tf.concat([tf.zeros([pad_count,1], tf.int32), stich_bank], 1)
   stich_bank_bool=tf.cast(stich_bank, tf.bool)
    
   pad_per_col=max_act-max_act_col
   pad_per_col=tf.expand_dims(pad_per_col,1)
    
   stich=tf.gather_nd(tf.transpose(stich_bank_bool),pad_per_col)
   stich=tf.transpose(stich)
   where_stich=tf.concat([where, stich], 0)
   act_set=tf.where(tf.transpose(where_stich))
   act_set = tf.expand_dims( act_set[:,1],1)
    
   act_set_bool = tf.transpose(where_stich)
   act_set_bool = tf.reshape(act_set_bool,[1,-1])
   act_set_rows=tf.where(act_set_bool)
   act_set_rows = tf.expand_dims( act_set_rows[:,1],1)
    
   DTD=tf.matmul(D,D, transpose_a=True) + (10**(-6) * tf.eye(tf.shape(D)[1],dtype=X.dtype))
   paddings=tf.stack([[0, pad_count], [0, pad_count]])
   DTD_paded=tf.pad(DTD,paddings)
    
   DTD_act = tf.gather_nd(DTD_paded, act_set)
   DTD_act = tf.reshape(DTD_act,[sample_count,-1,tf.shape(DTD_act)[1]])
   DTD_act = tf.transpose(DTD_act,[0,2,1])
   DTD_act = tf.reshape(DTD_act, [-1,tf.shape(DTD_act)[2]])
   DTD_act = tf.gather_nd(DTD_act, act_set_rows)
   DTD_act = tf.reshape(DTD_act,[-1,tf.shape(DTD_act)[1],tf.shape(DTD_act)[1]])
    
   paddings = tf.stack([[0, pad_count], [0, 0]])
   grad_output_paded = tf.pad(grad_output,paddings)
   grad_output_act = tf.gather_nd(tf.reshape(tf.transpose(grad_output_paded),[-1,1]), act_set_rows)
   grad_output_act = tf.reshape(grad_output_act,[sample_count,-1])
   grad_output_act = tf.expand_dims(grad_output_act,2)

   #s, u, v = tf.svd(DTD_act)
   #s_inv=tf.where(tf.less(s, 1e-20), s, 1./s)
   #s_inv=tf.expand_dims(s_inv,2)
   #Beta_act=tf.matmul(u, grad_output_act, transpose_a=True) 
   #Beta_act=tf.multiply(s_inv, Beta_act) 
   #Beta_act=tf.matmul(v, Beta_act) 
   #Beta_act= tf.squeeze(Beta_act)
    
   Beta_act=tf.matrix_solve_ls(DTD_act,grad_output_act,l2_regularizer=0, fast=False)
    
   shape=tf.stack([sample_count, tf.shape(grad_output_paded)[0]])
   shape=tf.cast(shape, tf.int64)
   ind=tf.where(tf.transpose(where_stich))
   Beta = tf.SparseTensor(ind, tf.reshape(Beta_act,[-1]), shape)
   Beta = tf.sparse_tensor_to_dense(Beta)
   Beta = tf.slice(Beta, [0,0], [sample_count,num_atoms]) 
   Beta = tf.transpose(Beta)
    
   lambda1_grad=-1*tf.reduce_sum(Beta) 
   X_grad_col=tf.matmul(D,Beta)  
    
   D_grad_tmp1=-1*tf.matmul(tf.matmul(D,Beta), A, transpose_b=True)
    
   E= X- tf.matmul(D,A)
   D_grad_tmp2=tf.matmul(E, Beta, transpose_b=True)
   D_grad=D_grad_tmp1+D_grad_tmp2
    
   return(lambda1_grad, X_grad_col, D_grad)


def col_to_image(X_col,ksizes,strides,rates,padding, target_h, target_w, target_dep, bsize):
   # I assumed rate=1 and square kernels
   filter_height=ksizes[1]
   filter_width=ksizes[2]
    
   if(padding=='SAME'):
      h_o = int(np.ceil(float(target_h) / float(strides[1])))
      w_o = int(np.ceil(float(target_w) / float(strides[2])))      
      if (target_h % strides[1] == 0):
         pad_along_height = max(filter_height - strides[1], 0)
      else:
         pad_along_height = max(filter_height - (target_h % strides[1]), 0)
      if (target_w % strides[2] == 0):
         pad_along_width = max(filter_width - strides[2], 0)
      else:
         pad_along_width = max(filter_width - (target_w % strides[2]), 0)

      pad_top = pad_along_height // 2
      pad_bottom = pad_along_height - pad_top
      pad_left = pad_along_width // 2
      pad_right = pad_along_width - pad_left
        
   else:
      h_o = int(np.ceil(float(target_h - filter_height + 1) / float(strides[1])))
      w_o  = int(np.ceil(float(target_w - filter_width + 1) / float(strides[2])))
      pad_top = 0
      pad_bottom = 0
      pad_left = 0
      pad_right = 0      
    
   #h_o=((target_h-ksize+(2*p))//stride)+1
   #print(X_col)
   X_col=tf.transpose(X_col)
    
   #print(X_col)
   X_col=tf.reshape(X_col,[bsize,h_o,w_o,-1]) #111
   #X_col=tf.transpose(X_col,[0,2,1,3])
   im_o = tf.zeros([bsize, target_h+pad_top+pad_bottom, target_w+pad_left+pad_right, target_dep],dtype=X_col.dtype)
   for j in range(h_o):
      for i in range(w_o):
         reg=tf.reshape(X_col[:,i,j,:],[-1,filter_height,filter_width,target_dep])
         index_i = i*strides[1]
         index_j = j*strides[2]
         pad=[[0,0],[index_i,target_h+pad_top+pad_bottom-(index_i+filter_height)]
                   ,[index_j,target_w+pad_left+pad_right-(index_j+filter_width)],[0,0]]
         paders=tf.pad(reg,pad)
         im_o+=paders
    
   if(padding=='SAME'):
      im_o=tf.slice(im_o, [0,pad_top,pad_left,0], [bsize,target_h,target_w,target_dep])
   return(im_o)



def col_to_image2(X_col,ksizes,strides,rates,padding, target_h, target_w, target_dep, bsize):
   # I assumed rate=1 and square kernels
   filter_height=ksizes[1]
   filter_width=ksizes[2]

   if(padding=='SAME'):
      h_o = int(np.ceil(float(target_h) / float(strides[1])))
      w_o = int(np.ceil(float(target_w) / float(strides[2])))
      if (target_h % strides[1] == 0):
         pad_along_height = max(filter_height - strides[1], 0)
      else:
         pad_along_height = max(filter_height - (target_h % strides[1]), 0)
      if (target_w % strides[2] == 0):
         pad_along_width = max(filter_width - strides[2], 0)
      else:
         pad_along_width = max(filter_width - (target_w % strides[2]), 0)
      pad_top = pad_along_height // 2
      pad_bottom = pad_along_height - pad_top
      pad_left = pad_along_width // 2
      pad_right = pad_along_width - pad_left

   else:
      h_o = int(np.ceil(float(target_h - filter_height + 1) / float(strides[1])))
      w_o  = int(np.ceil(float(target_w - filter_width + 1) / float(strides[2])))
      pad_top = 0
      pad_bottom = 0
      pad_left = 0
      pad_right = 0

    #h_o=((target_h-ksize+(2*p))//stride)+1
   X_col=tf.transpose(X_col)
   X_col=tf.reshape(X_col,[bsize,h_o,w_o,-1])
    
   idx = tf.where(tf.not_equal(X_col, 0))
   X_col_sp = tf.SparseTensor(idx, tf.gather_nd(X_col, idx), tf.cast(tf.shape(X_col), tf.int64))
   X_col_sp=tf.sparse_reshape(X_col_sp, [bsize,h_o,w_o,filter_height,filter_width,target_dep])
   idx = X_col_sp.indices
   idx_shifted =tf.concat([tf.expand_dims(idx[:,0],1), tf.expand_dims(idx[:,1],1), tf.expand_dims(idx[:,2],1),
                 tf.expand_dims(idx[:,1]*strides[1] + idx[:,3],1),
                 tf.expand_dims(idx[:,2]*strides[2] + idx[:,4],1), tf.expand_dims(idx[:,5],1)], 1)
    
   X_col_sp=tf.SparseTensor(idx_shifted, X_col_sp.values, [bsize, h_o, w_o, target_h+pad_top+pad_bottom, target_w+pad_left+pad_right,target_dep])
   im_o=tf.sparse_reduce_sum(X_col_sp, [1, 2])
    
    #im_o = tf.sparse_tensor_to_dense(X_col_sp)
    #im_o = tf.reduce_sum(im_o,[1,2])                  
    
   if(padding=='SAME'):
      im_o=tf.slice(im_o, [0,pad_top,pad_left,0], [bsize,target_h,target_w,target_dep])
   return(im_o)



def col_to_image3(X_col,ksizes,strides,rates,padding, target_h, target_w, target_dep, bsize):
   # I assumed rate=1 and square kernels
   filter_height=ksizes[1]
   filter_width=ksizes[2]

   if(padding=='SAME'):
      h_o = int(np.ceil(float(target_h) / float(strides[1])))
      w_o = int(np.ceil(float(target_w) / float(strides[2])))
      if (target_h % strides[1] == 0):
         pad_along_height = max(filter_height - strides[1], 0)
      else:
         pad_along_height = max(filter_height - (target_h % strides[1]), 0)
      if (target_w % strides[2] == 0):
         pad_along_width = max(filter_width - strides[2], 0)
      else:
         pad_along_width = max(filter_width - (target_w % strides[2]), 0)
      pad_top = pad_along_height // 2
      pad_bottom = pad_along_height - pad_top
      pad_left = pad_along_width // 2
      pad_right = pad_along_width - pad_left

   else:
      h_o = int(np.ceil(float(target_h - filter_height + 1) / float(strides[1])))
      w_o  = int(np.ceil(float(target_w - filter_width + 1) / float(strides[2])))
      pad_top = 0
      pad_bottom = 0
      pad_left = 0
      pad_right = 0

   #h_o=((target_h-ksize+(2*p))//stride)+1
    
   X_col=tf.transpose(X_col)
   X_col=tf.reshape(X_col,[bsize,h_o,w_o,-1])
   X_col=tf.reshape(X_col,[bsize,h_o*w_o,-1])

    
   idx_table= np.linspace(0,h_o,h_o,endpoint=False) *strides[1]
   idx_table= np.matlib.repmat(idx_table.astype(int),w_o,1)
   index_i = idx_table.reshape([1,-1])
    
   idx_table= np.linspace(0,w_o,w_o,endpoint=False) *strides[2]
   idx_table= np.matlib.repmat(idx_table.astype(int),h_o,1)
   index_j = idx_table.T.reshape([-1,1])
   
   idx= np.concatenate((index_j, index_i.T), axis=1)
        
   def cond(idx, X_col, output_ta, i, target_h, target_w, 
                                       pad_top, pad_bottom, pad_left, pad_right,
                                       filter_height, filter_width):
      return tf.less(i, idx.shape[0])

   def body(idx, X_col, output_ta, i, target_h, target_w, 
                                       pad_top, pad_bottom, pad_left, pad_right,
                                       filter_height, filter_width):
        
      reg=tf.reshape(X_col[:,i,:],[-1,filter_height,filter_width,target_dep])
      index_i = tf.cast(idx[i,0],tf.int32)
      index_j = tf.cast(idx[i,1],tf.int32)
        
        
      pad=tf.stack([[0,0],[index_i,target_h+pad_top+pad_bottom-(index_i+filter_height)]
                      ,[index_j,target_w+pad_left+pad_right-(index_j+filter_width)],[0,0]])
        
      reg=tf.pad(reg,pad)
      output_ta = output_ta.write(i, reg)
        
      return idx, X_col, output_ta, i + 1, target_h, target_w, pad_top, pad_bottom, pad_left, pad_right, filter_height, filter_width

   output_ta = tf.TensorArray(dtype=X_col.dtype, size=0, dynamic_size=True) 
   _, _, output_op, _, _,_, _,_,_,_,_,_  = tf.while_loop(cond, body, [idx, X_col, output_ta, 0, 
                                                     target_h, target_w, 
                                                     pad_top, pad_bottom, pad_left, pad_right,
                                                     filter_height, filter_width])
    
   im_o = output_op.stack()
   im_o = tf.reduce_sum(im_o,0 )
    
    
   #print(idx.shape)
   #print(idx)
                      
    
   if(padding=='SAME'):
      im_o=tf.slice(im_o, [0,pad_top,pad_left,0], [bsize,target_h,target_w,target_dep])
   return(im_o)


