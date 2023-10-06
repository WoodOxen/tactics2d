sudo docker run --rm \
	--name ubuntu2004 \
	-v /home/rowena/Documents/tactics2d:/tactics2d \
	ubuntu:20.04 \
	bash -c "chmod +x /tactics2d/scripts/pytest-ubuntu2004.sh && /tactics2d/scripts/pytest-ubuntu2004.sh"