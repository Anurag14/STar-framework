import numpy as np
from pdb import set_trace as bkpt

def return_index(args):
#   bkpt()
   if args.dataset == 'multi':
       if args.num==3:
           if args.target=='real':
               num_images=70358
           elif args.target=='painting':
               num_images=31502
           elif args.target=='clipart':
               num_images=18703
           else:
               num_images=24204
       else:
           if args.target=='real':
               num_images=70358
           elif args.target=='painting':
               num_images=31502
           elif args.target=='clipart':
               num_images=18703
           else:
               num_images=24456
   
   elif args.dataset=='office':
       if args.num==3:
           if args.target=='amazon':
               num_images=2817
       else:
           if args.target=='amazon':
               num_images=2817
   
   else:
       if args.num==3:
           if args.target=='Real':
               num_images=4357
           elif args.target=='Product':
               num_images=4439
           elif args.target=='Clipart':
               num_images=4365
           else:
               num_images=2427
       else:
           if args.target=='Real':
               num_images=4357
           elif args.target=='Product':
               num_images=4439
           elif args.target=='Clipart':
               num_images=4365
           else:
               num_images=2427			      
  
   return num_images
