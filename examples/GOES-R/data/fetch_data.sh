#!/bin/bash

# Copy data from the 21st of September at 18:00-19:00 GMT
aws s3 cp s3://noaa-goes17/ABI-L1b-RadF/2021/264/18/ noaa-goes17/ABI-L1b-RadF/2021/264/18 --recursive --no-sign-request
aws s3 cp s3://noaa-goes17/GLM-L2-LCFA/2021/264/18/ noaa-goes17/GLM-L2-LCFA/2021/264/18/ --recursive --no-sign-request
aws s3 cp s3://noaa-goes17/ABI-L2-LSTF/2021/264/18/ data/noaa-goes17/ABI-L2-LSTF/2021/264/18 --recursive --no-sign-request
