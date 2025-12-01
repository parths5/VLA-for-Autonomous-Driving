"""
Utility functions for parsing Bench2Drive dataset annotations.
"""

import json
import gzip
import numpy as np
from typing import Dict, List, Tuple, Optional


def load_annotation(anno_path: str) -> Dict:
    """
    Load a single annotation file (JSON.gz format).
    
    Args:
        anno_path: Path to annotation file
        
    Returns:
        Dictionary containing annotation data
    """
    with gzip.open(anno_path, 'rt') as f:
        return json.load(f)


def get_ego_transform(anno: Dict) -> Tuple[float, float, float]:
    """
    Extract ego vehicle position and orientation from annotation.
    
    Args:
        anno: Annotation dictionary
        
    Returns:
        Tuple of (x, y, theta) in world coordinates
    """
    return anno['x'], anno['y'], anno['theta']


def get_ego_speed(anno: Dict) -> float:
    """
    Extract ego vehicle speed from annotation.
    
    Args:
        anno: Annotation dictionary
        
    Returns:
        Speed in m/s
    """
    return anno['speed']


def get_control_commands(anno: Dict) -> Dict[str, float]:
    """
    Extract control commands (throttle, steer, brake).
    
    Args:
        anno: Annotation dictionary
        
    Returns:
        Dictionary with throttle, steer, brake values
    """
    return {
        'throttle': anno['throttle'],
        'steer': anno['steer'],
        'brake': anno['brake'],
    }


def get_navigation_command(anno: Dict) -> Tuple[int, int]:
    """
    Extract navigation commands.
    
    Bench2Drive command mapping:
    1: Turn left
    2: Turn right
    3: Go straight
    4: Follow lane
    5: Lane change left
    6: Lane change right
    
    Args:
        anno: Annotation dictionary
        
    Returns:
        Tuple of (command_near, command_far)
    """
    return anno['command_near'], anno['command_far']


def get_target_point(anno: Dict) -> Tuple[float, float]:
    """
    Extract target waypoint in ego-centric coordinates.
    
    Args:
        anno: Annotation dictionary
        
    Returns:
        Tuple of (x_target, y_target) relative to ego vehicle
    """
    return anno['x_target'], anno['y_target']


def world_to_ego(world_x: float, world_y: float, 
                 ego_x: float, ego_y: float, ego_theta: float) -> Tuple[float, float]:
    """
    Convert world coordinates to ego-centric coordinates.
    
    Args:
        world_x, world_y: Point in world coordinates
        ego_x, ego_y: Ego vehicle position in world coordinates
        ego_theta: Ego vehicle orientation (radians)
        
    Returns:
        Tuple of (x, y) in ego-centric coordinates
    """
    # Translate to ego origin
    dx = world_x - ego_x
    dy = world_y - ego_y
    
    # Rotate to ego frame (ego vehicle points in +x direction)
    cos_theta = np.cos(-ego_theta)
    sin_theta = np.sin(-ego_theta)
    
    ego_x_local = cos_theta * dx - sin_theta * dy
    ego_y_local = sin_theta * dx + cos_theta * dy
    
    return ego_x_local, ego_y_local


def compute_waypoints_from_trajectory(
    annotations: List[Dict],
    current_idx: int,
    num_waypoints: int = 11,
    waypoint_spacing: float = 0.2,
    fps: float = 10.0
) -> np.ndarray:
    """
    Compute future waypoints from trajectory annotations.
    
    Args:
        annotations: List of annotation dictionaries for the sequence
        current_idx: Index of current frame
        num_waypoints: Number of waypoints to predict
        waypoint_spacing: Time spacing between waypoints (seconds)
        fps: Frames per second of the dataset
        
    Returns:
        Array of shape (num_waypoints, 2) containing waypoints in ego-centric coordinates
    """
    # Calculate frame spacing
    frame_spacing = int(waypoint_spacing * fps)
    
    # Get current ego pose
    current_anno = annotations[current_idx]
    ego_x, ego_y, ego_theta = get_ego_transform(current_anno)
    
    waypoints = []
    for i in range(1, num_waypoints + 1):
        future_idx = current_idx + i * frame_spacing
        
        if future_idx < len(annotations):
            future_anno = annotations[future_idx]
            future_x, future_y, _ = get_ego_transform(future_anno)
            
            # Convert to ego-centric coordinates
            wp_x, wp_y = world_to_ego(future_x, future_y, ego_x, ego_y, ego_theta)
            waypoints.append([wp_x, wp_y])
        else:
            # If we run out of future frames, repeat the last waypoint
            if waypoints:
                waypoints.append(waypoints[-1])
            else:
                waypoints.append([0.0, 0.0])
    
    return np.array(waypoints, dtype=np.float32)


def get_bounding_boxes(anno: Dict, ego_x: float, ego_y: float, ego_theta: float) -> List[Dict]:
    """
    Extract and convert bounding boxes to ego-centric coordinates.
    
    Args:
        anno: Annotation dictionary
        ego_x, ego_y: Ego vehicle position
        ego_theta: Ego vehicle orientation
        
    Returns:
        List of bounding box dictionaries with ego-centric coordinates
    """
    boxes = []
    for bbox in anno.get('bounding_boxes', []):
        if bbox['class'] == 'ego_vehicle':
            continue
            
        # Get center location
        center = bbox['center']
        
        # Convert to ego-centric
        box_x, box_y = world_to_ego(center[0], center[1], ego_x, ego_y, ego_theta)
        
        boxes.append({
            'class': bbox['class'],
            'type_id': bbox.get('type_id', ''),
            'position': [box_x, box_y, center[2]],  # Keep z as is
            'extent': bbox['extent'],
            'rotation': bbox['rotation'],
        })
    
    return boxes


def map_command_to_simlingo(bench2drive_cmd: int) -> int:
    """
    Map Bench2Drive command to SimLingo format.
    
    Bench2Drive: 1=left, 2=right, 3=straight, 4=follow, 5=lane_left, 6=lane_right
    SimLingo: 1=left, 2=right, 3=straight, 4=follow, 5=lane_left, 6=lane_right
    
    Args:
        bench2drive_cmd: Command from Bench2Drive
        
    Returns:
        Command in SimLingo format (same mapping)
    """
    # Commands are already compatible
    return bench2drive_cmd


def get_weather_info(anno: Dict) -> Dict:
    """
    Extract weather information.
    
    Args:
        anno: Annotation dictionary
        
    Returns:
        Dictionary with weather parameters
    """
    return anno.get('weather', {})

