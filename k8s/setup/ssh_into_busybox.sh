kubectl replace --force -f setup/busybox.yaml
sleep 10
kubectl exec --stdin --tty busybox -- sh