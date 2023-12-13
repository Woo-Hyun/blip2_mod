import numpy as np
from nuscenes.prediction import (PredictHelper,
                                 convert_local_coords_to_global,
                                 convert_global_coords_to_local)
#from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
# from mmcv.parallel import DataContainer as DC
#from mmdet.datasets.pipelines import to_tensor

class NuScenesTraj(object):
    def __init__(self,
                 nusc,
                 predict_steps,
                 past_steps,
                 use_nonlinear_optimizer=False):
        super().__init__()
        self.nusc = nusc
        self.predict_steps = predict_steps
        self.past_steps = past_steps

        self.predict_helper = PredictHelper(self.nusc)
        self.use_nonlinear_optimizer = use_nonlinear_optimizer

    def get_traj_label(self, sample_token, ann_tokens):
        sd_rec = self.nusc.get('sample', sample_token)
        fut_traj_all = []
        fut_traj_valid_mask_all = []
        past_traj_all = []	
        past_traj_valid_mask_all = []

        distance_all = []
        agent_type = []
        Velocity_list = []
        Acceleration_list = []
        Heading_list = []
        sd_record = self.nusc.get('sample_data', sd_rec['data']['LIDAR_TOP'])
        pose_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
        av_translation = pose_record['translation']
        av_rot = pose_record['rotation']
        av_pos = pose_record['translation'][:2]
        _, boxes, _ = self.nusc.get_sample_data(sd_rec['data']['LIDAR_TOP'], selected_anntokens=ann_tokens)
        for i, ann_token in enumerate(ann_tokens):
            box = boxes[i]
            instance_token = self.nusc.get('sample_annotation', ann_token)['instance_token']
            fut_traj_local = self.predict_helper.get_future_for_agent(instance_token, sample_token, seconds=6, in_agent_frame=False)
            past_traj_local = self.predict_helper.get_past_for_agent(instance_token, sample_token, seconds=2, in_agent_frame=True)
            global_past_traj = self.predict_helper.get_past_for_agent(instance_token, sample_token, seconds=2, in_agent_frame=False)
            annotation = self.predict_helper.get_sample_annotation(instance_token, sample_token)
            curr_state = annotation['translation'][:2]


            my_annotation_metadata =  self.nusc.get('sample_annotation', ann_token)
            agent_type.append(my_annotation_metadata['category_name'])

            # Meters / second.

            Velocity = self.predict_helper.get_velocity_for_agent(instance_token, sample_token)
            Acceleration = self.predict_helper.get_acceleration_for_agent(instance_token, sample_token)
            Heading = self.predict_helper.get_heading_change_rate_for_agent(instance_token, sample_token)
            Velocity_list.append(Velocity)
            Acceleration_list.append(Acceleration)
            Heading_list.append(Heading)

            fut_traj = np.zeros((self.predict_steps, 2))
            fut_traj_valid_mask = np.zeros((self.predict_steps, 2))
            past_traj = np.zeros((self.past_steps+1, 2))		
            past_traj_valid_mask = np.zeros((self.past_steps+1, 2))
            distance = np.zeros((self.past_steps, 2))

            if fut_traj_local.shape[0] > 0:
                if self.use_nonlinear_optimizer:
                    trans = box.center
                else:
                    trans = np.array([0, 0, 0])
                rot = Quaternion(matrix=box.rotation_matrix)
                # fut_traj_scence_centric = convert_local_coords_to_global(fut_traj_local, trans, rot) 
                fut_traj[:fut_traj_local.shape[0], :] = convert_global_coords_to_local(fut_traj_local,av_translation,av_rot)  
                fut_traj_valid_mask[:fut_traj_local.shape[0], :] = 1
            if past_traj_local.shape[0] > 0:	
                trans = np.array([0, 0, 0])		
                rot = Quaternion(matrix=box.rotation_matrix)		
                distance[:past_traj_local.shape[0], :] = convert_global_coords_to_local(global_past_traj,av_translation,av_rot)
                # past_traj_scence_centric = convert_local_coords_to_global(past_traj_local, trans, rot) 		
                past_traj[1:global_past_traj.shape[0]+1, :] = convert_global_coords_to_local(global_past_traj,av_translation,av_rot)
                past_traj_valid_mask[1:global_past_traj.shape[0]+1, :] = 1
            past_traj[0,:] = convert_global_coords_to_local(curr_state,av_translation,av_rot)
            past_traj_valid_mask[0, :] = 1
            fut_traj_all.append(fut_traj)		
            fut_traj_valid_mask_all.append(fut_traj_valid_mask)		
            past_traj_all.append(past_traj)		
            past_traj_valid_mask_all.append(past_traj_valid_mask)		
            distance_all.append(distance)
        if len(ann_tokens) > 0:		
            fut_traj_all = np.stack(fut_traj_all, axis=0)		
            fut_traj_valid_mask_all = np.stack(fut_traj_valid_mask_all, axis=0)		
            past_traj_all = np.stack(past_traj_all, axis=0)		
            past_traj_valid_mask_all = np.stack(past_traj_valid_mask_all, axis=0)		
            distance_all = np.stack(distance_all, axis=0)
        else:		
            fut_traj_all = np.zeros((0, self.predict_steps, 2))		
            fut_traj_valid_mask_all = np.zeros((0, self.predict_steps, 2))		
            past_traj_all = np.zeros((0, self.past_steps+1, 2))		
            past_traj_valid_mask_all = np.zeros((0, self.past_steps+1, 2))	
            distance_all = np.zeros((0, self.past_steps, 2))

        if Velocity_list != []:
            return fut_traj_all, fut_traj_valid_mask_all, past_traj_all, past_traj_valid_mask_all,distance_all,np.stack(agent_type,axis=0),np.stack(Velocity_list,axis=0),np.stack(Acceleration_list,axis=0),np.stack(Heading_list,axis=0), np.array(av_pos)
        else:
            print("velocity list is empty")
            return fut_traj_all, fut_traj_valid_mask_all, past_traj_all, past_traj_valid_mask_all,distance_all,np.array(agent_type), np.array(Velocity_list),np.array(Acceleration_list),np.array(Heading_list), np.array(av_pos)
    
    def get_traj_label_single_instance(self, sample_token, ann_tokens, instance_tok):
        sd_rec = self.nusc.get('sample', sample_token)
        fut_traj_all = []
        fut_traj_valid_mask_all = []
        past_traj_all = []	
        past_traj_valid_mask_all = []
        bev_pos = []

        distance_all = []
        agent_type = []
        Velocity_list = []
        Acceleration_list = []
        Heading_list = []
        sd_record = self.nusc.get('sample_data', sd_rec['data']['LIDAR_TOP'])
        pose_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
        av_translation = pose_record['translation']
        av_rot = pose_record['rotation']
        av_pos = pose_record['translation'][:2]
        _, boxes, _ = self.nusc.get_sample_data(sd_rec['data']['LIDAR_TOP'], selected_anntokens=ann_tokens)
        for i, ann_token in enumerate(ann_tokens):
            box = boxes[i]
            instance_token = instance_tok
            fut_traj_local = self.predict_helper.get_future_for_agent(instance_token, sample_token, seconds=6, in_agent_frame=True)
            past_traj_local = self.predict_helper.get_past_for_agent(instance_token, sample_token, seconds=2, in_agent_frame=True)
            global_past_traj = self.predict_helper.get_past_for_agent(instance_token, sample_token, seconds=2, in_agent_frame=True)
            annotation = self.predict_helper.get_sample_annotation(instance_token, sample_token)
            curr_state = annotation['translation'][:2]


            my_annotation_metadata =  self.nusc.get('sample_annotation', ann_token)
            agent_type.append(my_annotation_metadata['category_name'])

            # Meters / second.

            Velocity = self.predict_helper.get_velocity_for_agent(instance_token, sample_token)
            Acceleration = self.predict_helper.get_acceleration_for_agent(instance_token, sample_token)
            Heading = self.predict_helper.get_heading_change_rate_for_agent(instance_token, sample_token)
            Velocity_list.append(Velocity)
            Acceleration_list.append(Acceleration)
            Heading_list.append(Heading)

            fut_traj = np.zeros((self.predict_steps, 2))
            fut_traj_valid_mask = np.zeros((self.predict_steps, 2))
            past_traj = np.zeros((self.past_steps+1, 2))		
            past_traj_valid_mask = np.zeros((self.past_steps+1, 2))
            distance = np.zeros((self.past_steps, 2))

            if fut_traj_local.shape[0] > 0:
                if self.use_nonlinear_optimizer:
                    trans = box.center
                else:
                    trans = np.array([0, 0, 0])
                rot = Quaternion(matrix=box.rotation_matrix)
                # # fut_traj_scence_centric = convert_local_coords_to_global(fut_traj_local, trans, rot) 
                # fut_traj[:fut_traj_local.shape[0], :] = convert_global_coords_to_local(fut_traj_local,av_translation,av_rot)
                fut_traj[:fut_traj_local.shape[0], :] = fut_traj_local  
                fut_traj_valid_mask[:fut_traj_local.shape[0], :] = 1
            if past_traj_local.shape[0] > 0:	
                trans = np.array([0, 0, 0])		
                rot = Quaternion(matrix=box.rotation_matrix)		
                distance[:past_traj_local.shape[0], :] = convert_global_coords_to_local(global_past_traj,av_translation,av_rot)
                # distance[:past_traj_local.shape[0], :] = global_past_traj
                # # past_traj_scence_centric = convert_local_coords_to_global(past_traj_local, trans, rot) 		
                past_traj[1:global_past_traj.shape[0]+1, :] = convert_global_coords_to_local(global_past_traj,av_translation,av_rot)
                # past_traj[1:global_past_traj.shape[0]+1, :] = global_past_traj
                past_traj_valid_mask[1:global_past_traj.shape[0]+1, :] = 1
            past_traj[0,:] = convert_global_coords_to_local(curr_state,av_translation,av_rot)
            bev_pos.append(convert_global_coords_to_local(curr_state,av_translation,av_rot))
            past_traj[0,:] = 0.0
            past_traj_valid_mask[0, :] = 1
            fut_traj_all.append(fut_traj)		
            fut_traj_valid_mask_all.append(fut_traj_valid_mask)		
            past_traj_all.append(past_traj)		
            past_traj_valid_mask_all.append(past_traj_valid_mask)
            bev_pos.append(convert_global_coords_to_local(curr_state,av_translation,av_rot))
            break
        # if len(ann_tokens) > 0:		
        #     fut_traj_all = np.stack(fut_traj_all, axis=0)		
        #     fut_traj_valid_mask_all = np.stack(fut_traj_valid_mask_all, axis=0)		
        #     past_traj_all = np.stack(past_traj_all, axis=0)		
        #     past_traj_valid_mask_all = np.stack(past_traj_valid_mask_all, axis=0)		
        #     distance_all = np.stack(distance_all, axis=0)
        # else:		
        #     fut_traj_all = np.zeros((0, self.predict_steps, 2))		
        #     fut_traj_valid_mask_all = np.zeros((0, self.predict_steps, 2))		
        #     past_traj_all = np.zeros((0, self.past_steps+1, 2))		
        #     past_traj_valid_mask_all = np.zeros((0, self.past_steps+1, 2))	
        #     distance_all = np.zeros((0, self.past_steps, 2))

        if Velocity_list != []:
            return fut_traj_all, fut_traj_valid_mask_all, past_traj_all, past_traj_valid_mask_all, bev_pos, distance_all,np.stack(agent_type,axis=0),np.stack(Velocity_list,axis=0),np.stack(Acceleration_list,axis=0),np.stack(Heading_list,axis=0), np.array(av_pos)
        else:
            print("velocity list is empty")
            return fut_traj_all, fut_traj_valid_mask_all, past_traj_all, past_traj_valid_mask_all, bev_pos, distance_all,np.array(agent_type), np.array(Velocity_list),np.array(Acceleration_list),np.array(Heading_list), np.array(av_pos)