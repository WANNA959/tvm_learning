from tvm.driver import tvmc

model = tvmc.load('resnet50-v2-7.onnx')  # Step 1: Load

tvmc.tune(model, target="llvm")  # Step 1.5: Optional Tune

package = tvmc.compile(model, target="llvm")  # Step 2: Compile

result = tvmc.run(package, device="cpu")  # Step 3: Run
print(result)
