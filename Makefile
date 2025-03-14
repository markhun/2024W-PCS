dev-container:
	docker build -t markhun/cuda-dev-image .
	docker run -d -it --gpus all --name dev -p 2221:22 -v "$$(pwd)":/home/dev/commbench $$(docker build -q .) && \
	docker container list
	@echo "SSH into the listed port. User: 'dev' Password: 'pass'"

# dev-container:
# 	docker run --gpus all -d -it --name dev -p 2221:22 -v "$$(pwd)":/home/dev/commbench $$(docker build -q .) && \
# 	docker container list

setup-chameleon-cuda2404-container:
	sudo apt-get install -y libmpich-dev libopenmpi-dev
	sudo apt-get install --no-install-recommends -y libnccl2 libnccl-dev