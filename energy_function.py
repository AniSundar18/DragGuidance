import torch
import torch.nn.functional as Func

def style_loss(cls_tgt, cls, tgt_img, img):
    L2_loss = torch.nn.MSELoss()
    return L2_loss(cls, cls_tgt)  



def get_cosine_similarity_map(F_gen, q):
    # F_gen: dim x h x w
    # q : dim
    f_gen = F_gen.view(F_gen.shape[0], -1)
    cosine_similarities = Func.cosine_similarity(q.unsqueeze(0), f_gen.T, dim=1)
    similarity_heatmap = cosine_similarities.view(F_gen.shape[1:])
    return similarity_heatmap

def get_centroid(cosine_similarity_map):
    print('Shape: ', cosine_similarity_map.shape)
    normalized_map = cosine_similarity_map / torch.sum(cosine_similarity_map)
    rows, cols = torch.meshgrid(torch.arange(64), torch.arange(64))
    rows = rows.to('cuda')
    cols = cols.to('cuda')
    # Calculate weighted sum of row and column indices
    weighted_row_sum = torch.sum(rows.float() * normalized_map)
    weighted_col_sum = torch.sum(cols.float() * normalized_map)

    # Calculate the total sum of normalized similarity values
    total_similarity_sum = torch.sum(normalized_map)

    # Calculate the centroid
    centroid_row = weighted_row_sum / total_similarity_sum
    centroid_col = weighted_col_sum / total_similarity_sum
    return torch.cat((centroid_row.unsqueeze(0) , centroid_col.unsqueeze(0)))

def get_appearence(F):
    average_vector = torch.mean(F, dim=(1, 2))
    return average_vector

def E_appearence(A_gen, A_guid):
    loss = 0
    L2_loss = torch.nn.MSELoss()
    L1_loss = torch.nn.L1Loss()
    for key in A_gen.keys():
        if key in ['2', '3']:
            loss += L1_loss(A_gen[key], A_guid[key])
    return loss

def E_size(F_guid, F_gen, opt):
    pass

def E_movement(F_guid, F_gen, opt, timestep = 200):
    L1_loss = torch.nn.L1Loss()
    L2_loss = torch.nn.MSELoss()
    loss = None
    handle_points = opt.handle_points
    target_points = opt.target_points
    f_gen = F_gen['2'][0]
    f_guid = F_guid['2'][0]
    f_guid.detach()
    for idx in range(len(handle_points)):
        scale = None
        hi = torch.Tensor(list(handle_points[idx]))/8
        ti = torch.Tensor(list(target_points[idx]))/8
        q = f_guid[:,int(hi[0].item()), int(hi[1].item())]
        similarity_map = get_cosine_similarity_map(f_gen, q)
        similarity_map_guid = get_cosine_similarity_map(f_guid, q)
        #Thresholding needed
        similarity_map = torch.sigmoid(25 * (similarity_map - 0.70))
        similarity_map_guid = torch.sigmoid(25 * (similarity_map_guid - 0.70))
        #Object part Size and appearance
        a_gen = similarity_map * F_gen['2'][0]
        a_guid = similarity_map_guid * F_guid['2'][0]
        #size_gen = torch.sum(similarity_map)/(f_gen.shape[1] * f_gen.shape[2])
        #size_guid = torch.sum(similarity_map_guid)/(f_guid.shape[1] * f_guid.shape[2])
        #Disentangle Appearance
        A_gen = {}
        A_guid = {}
        #Image level Appearance
        for key in F_guid.keys():
            A_gen[key] = get_appearence(F_gen[key][0])
            A_guid[key] = get_appearence(F_guid[key][0])

        #Object level appearance
        obj_app_gen = torch.sum(a_gen, dim = 0)/torch.sum(similarity_map)
        obj_app_guid = torch.sum(a_guid, dim = 0)/torch.sum(similarity_map_guid)
        centroid = get_centroid(similarity_map)
        k = torch.round(centroid)
        if True:
            print(centroid, ti)
            E_app = E_appearence(A_gen = A_gen, A_guid = A_guid)
            print('E_app: ', E_app) 
            E_obj_app = L2_loss(obj_app_gen, obj_app_guid)
            print('E_obj_app: ', E_obj_app) 
            #E_size = L1_loss(size_guid, size_gen)
            #print('E_size: ', E_size)
            if loss == None:
                
                loss = L1_loss(centroid, ti.to(opt.device)) + 10 * E_app #+ 1* E_obj_app
            else :
                loss += L1_loss(centroid, ti.to(opt.device)) + 10 * E_app #+ 1 * E_obj_app
            scale = loss.detach()/L1_loss(hi.to(opt.device), ti.to(opt.device)) 
            print('Scale: ', scale)
    return loss #* scale.item()




       
