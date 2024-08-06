import subprocess

main_return = 0
while main_return ==0:
    proccess = subprocess.run(["python", "aspen_win32_.py"], capture_output = True, text =True)
    main_return = proccess.returncode
    print(main_return)
