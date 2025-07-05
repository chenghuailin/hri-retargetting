# 2025.04.21 HIT-xiaowangzi
# Print the collsion links 

from pathlib import Path
 
import os
import pinocchio as pin
from HRI_retarget import DATA_ROOT
import numpy as np
import time
 
pinocchio_model_dir = os.path.join(DATA_ROOT,"resources/robots/g1")
 
model_path = os.path.join(DATA_ROOT,"resources/robots/g1")
mesh_dir = os.path.join(DATA_ROOT,"resources/robots/g1/meshes")
urdf_filename = "g1_29dof_rev_1_0.urdf"
urdf_model_path = os.path.join(model_path, urdf_filename)
 
# Load model
model = pin.buildModelFromUrdf(urdf_model_path)
 
# Load collision geometries
geom_model = pin.buildGeomFromUrdf(
    model, urdf_model_path, pin.GeometryType.COLLISION, mesh_dir
)
 
# Add collisition pairs
geom_model.addAllCollisionPairs()
print("num collision pairs - initial:", len(geom_model.collisionPairs))
 
# Remove collision pairs listed in the SRDF file
srdf_filename = "g1_29dof_rev_1_0.srdf"
srdf_model_path = os.path.join(model_path, srdf_filename)
 
pin.removeCollisionPairs(model, geom_model, srdf_model_path)
print(
    "num collision pairs - after removing useless collision pairs:",
    len(geom_model.collisionPairs),
)
 
# Load reference configuration
pin.loadReferenceConfigurations(model, srdf_model_path)
 
# Retrieve the half sitting position from the SRDF file
# q = model.referenceConfigurations["half_sitting"]
# q = np.zeros(model.njoints)
print(f"Total joints in Pinocchio model: {model.njoints}")
for i, joint in enumerate(model.joints):
    print(f"Joint {i}: {model.names[i]}")

q = pin.randomConfiguration(model)
print("random q:", q)

# Create data structures
data = model.createData()
geom_data = pin.GeometryData(geom_model)

 
 
# Compute all the collisions
pin.computeCollisions(model, data, geom_model, geom_data, q, False)
 
# Print the status of collision for all collision pairs
for k in range(len(geom_model.collisionPairs)):
    cr = geom_data.collisionResults[k]
    cp = geom_model.collisionPairs[k]
    
    # name1 = geom_model.geometryObjects[cp.first].name
    # name2 = geom_model.geometryObjects[cp.second].name
        # 原始 mesh 名称
    geo1 = geom_model.geometryObjects[cp.first]
    geo2 = geom_model.geometryObjects[cp.second]

    # 真正的 link 名称（无 _0/_1 后缀）
    link1 = model.frames[geo1.parentFrame].name
    link2 = model.frames[geo2.parentFrame].name
    
    print(
        "collision pair:",
        {link1},
        ",",
        {link2},
        "- collision:",
        "Yes" if cr.isCollision() else "No",
    )
 
# Compute for a single pair of collision
pin.updateGeometryPlacements(model, data, geom_model, geom_data, q)
pin.computeCollision(geom_model, geom_data, 0)
