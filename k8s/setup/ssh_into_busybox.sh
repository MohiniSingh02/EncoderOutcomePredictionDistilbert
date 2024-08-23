kubectl replace --force -f setup/busybox.yaml
sleep 10
winpty kubectl exec --stdin --tty busybox -- sh