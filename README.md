## Workflows

Download [FastMRI](https://fastmri.med.nyu.edu/) dataset, configure the dataset root in the promptmr, resnet, and eddpg. [PromptMR](https://github.com/hellopipu/PromptMR) follows its own repo, no extra setup required. You should first run the initial setup script. Make it executable and then run it with the following command:
```
chmod +x setup.sh && ./setup.sh
```
After the initial setup, you can train the models using their corresponding shell scripts. To train ResNet, execute:
```
./run/train_resnet.sh
```
To train E-DDPG, execute:
```
./run/train_eddpg.sh
```
The testing process is similar to training. Use the provided testing scripts located in the same directory. For detailed instructions and options, please refer to the comments inside each individual shell (.sh) file.

