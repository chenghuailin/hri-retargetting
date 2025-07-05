from pathlib import Path
import os
import numpy as np
 
import pinocchio as pin
from HRI_retarget import DATA_ROOT
 
pinocchio_model_dir = os.path.join(DATA_ROOT,"resources/robots/g1")
 
model_path = os.path.join(DATA_ROOT,"resources/robots/g1")
mesh_dir = os.path.join(DATA_ROOT,"resources/robots/g1/meshes")
urdf_filename = "g1_29dof_rev_1_0.urdf"
urdf_model_path = os.path.join(model_path, urdf_filename)
 
# Load model
model  = pin.buildModelFromUrdf(urdf_model_path, pin.JointModelFreeFlyer(), mesh_dir)
 
# Load collision geometries
geom_model = pin.buildGeomFromUrdf(
    model, urdf_model_path, pin.GeometryType.COLLISION
)
 
# Add collisition pairs
geom_model.addAllCollisionPairs()
print(dir(model))
print(model.frames)

print("num collision pairs - initial:", len(geom_model.collisionPairs))
# Remove collision pairs listed in the SRDF file
srdf_filename = "g1_29dof_rev_1_0.srdf"
srdf_model_path = os.path.join(model_path , srdf_filename)
 
pin.removeCollisionPairs(model, geom_model, srdf_model_path)
print(
    "num collision pairs - after removing useless collision pairs:",
    len(geom_model.collisionPairs),
)
 
# # # Load reference configuration
# # pin.loadReferenceConfigurations(model, srdf_model_path)
 
# # # Retrieve the half sitting position from the SRDF file
# # q = model.referenceConfigurations["half_sitting"]


q = np.array([0,0,0.785,0,0,0,1,
                        -0.15,0,0,0.3,-0.15,0,
                        -0.15,0,0,0.3,-0.15,0,
                        0,0,0,
                        0, 1.57,0,1.57,0,0,0,
                        0,-1.57,0,1.57,0,0,0]).astype(np.float32)
# q = np.zeros(36)
 
# Create data structures
data = model.createData()
geom_data = pin.GeometryData(geom_model)
 
 
# Compute all the collisions
pin.computeCollisions(model, data, geom_model, geom_data, q, False)
 
# Print the status of collision for all collision pairs
for k in range(len(geom_model.collisionPairs)):
    cr = geom_data.collisionResults[k]
    cp = geom_model.collisionPairs[k]
    if not cr.isCollision():continue
    print(
        "collision pair:",
        cp.first,
        ",",
        cp.second,
        "- collision:",
        "Yes" if cr.isCollision() else "No",
    )

# import ipdb 
# ipdb.set_trace()
 
# Compute for a single pair of collision
pin.updateGeometryPlacements(model, data, geom_model, geom_data, q)
flag = pin.computeCollision(geom_model, geom_data, 0)
print(flag)

