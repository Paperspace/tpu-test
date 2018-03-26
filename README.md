# Gradient tpu-test

To run a tpu test gradient job:

1. git clone git@github.com:Paperspace/tpu-test.git
2. cd tpu-test
3. paperspace project init
4. paperspace jobs create --machineType TPU --container gcr.io/tensorflow/tensorflow:1.6.0 --command "python main.py"
