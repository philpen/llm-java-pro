
# llm-java-pro

Implementation of the GPT2 Large Language Model (LLM) example in Java. This is a port of the original project (llm.c) found [here](https://github.com/karpathy/llm.c). 

## Preparation for Running ChatGPT2 in Java

It is necessary to undertake some preparatory steps before running. These steps mirror those found in the original [llm.c repository](https://github.com/karpathy/llm.c). Despite the presence of the same code in this repository, bear in mind that LLM.c is an ongoing project.

It is strongly recommended to run the original llm.c to gain perspective of its functioning.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python prepro_tinyshakespeare.py
python train_gpt2.py
```

### JVM Requirements

This version runs on GraalVM 21. If you're using sdkman, it is necessary to use this command:

```bash
sdk default java 21.0.2-graalce
```

The operation of the following JVM versions have been tested and found to be functional. However, please note that some versions run slower than the others.

1. Temurin: This ran at half the speed of GraalVM. Terminated it after step 10

```bash
sdk install java 21-tem
sdk use java 21-tem
```

2. Correto: This VM was also slower than GraalVM. Hence, terminated it after step 10

```bash
sdk install java 21.0.3-amzn
sdk use java 21.0.3-amzn
```

## Executing the Project

The arguments passed to the JVM are important, namely "-Djava.util.concurrent.ForkJoinPool.common.parallelism=10" should be adjusted based on how many cores are available. As matrix multiplication methods are CPU bound, adding more threads than cores will be counterproductive. 

```bash
mvn clean install;
java -jar -ea --add-modules jdk.incubator.vector --enable-preview -Xmx8g -Djava.util.concurrent.ForkJoinPool.common.parallelism=10 target/gpt2-1.0-SNAPSHOT.jar
```

## Performance

While specific performance tuning hasn't been undertaken, it should be noted that the C version currently operates faster than this version. There are potential areas for improvement such as parallelizing some of the loops. The matmul_forward and matmul_backward methods have been made parallel to prevent excessive slowdowns.