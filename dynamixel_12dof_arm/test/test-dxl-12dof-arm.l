#+:ros (require :unittest "lib/llib/unittest.l")

(unless *unit-test*
  (init-unit-test))

(deftest test-init-model
  (load "package://dynamixel_12dof_arm/euslisp/dxl-12dof-arm-robot.l")
  (assert (setq *dxl-12dof-arm* (dxl-12dof-arm)))
  (assert (objects (list *dxl-12dof-arm*)))
  (dolist (pose (remove-if-not #'(lambda (x) (substringp "pose" (string-downcase x))) (send *dxl-12dof-arm* :methods)))
    (assert (send *dxl-12dof-arm* pose)))
  (send *dxl-12dof-arm* :tuckarm-pose)
  (assert (send *dxl-12dof-arm* :arm :move-end-pos (float-vector 20 20 -20) :world :rotation-axis nil))
  )

(deftest test-init-ri
  (load "package://dynamixel_12dof_arm/euslisp/dxl-12dof-arm-interface.l")
  (assert (dxl-12dof-arm-init))
  (assert (progn
            (send *ri* :angle-vector (send *dxl-12dof-arm* :angle-vector) 2000)
            (send *ri* :wait-interpolation)
            t))
  )

(run-all-tests)
(exit 0)
