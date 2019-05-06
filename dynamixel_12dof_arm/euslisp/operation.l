;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; 必要なファイルのロードとROSのtopicのsubscribe/advertise設定
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(ros::load-ros-manifest "dynamixel_msgs") ;; dynamixelのメッセージを用いるためのload設定
(ros::roseus "test_arm") ;; ros initを行う

;; 7自由度分いっぺんに動かすサンプルになっているが、
;; いずれも1軸ずつ動かすことができる
(defvar *arm-dof* 12)
(setq *state-list* (make-list *arm-dof*))

;; subscribeの設定
;;   どのtopicをsubscribeするか (センサ値などの取得用)

;;   dynamixel_msgs::JointState型のメッセージをsubscribeするときに呼ばれるコールバック関数
;;     具体的には/arm_j0_controllder/stateなどをsubscribeすることになる
(defun dxl-state-cb
  (msg)
  (setf (elt *state-list* (1- (read-from-string (string-left-trim "arm_joint" (send msg :name))))) msg))

;;   subscriberの登録
(dotimes (i *arm-dof*)
  (ros::subscribe (format nil "/arm_j~d_controller/state" (1+ i)) ;; subscribeしたいtopic名
                  dynamixel_msgs::JointState ;; topicの型名
                  #'dxl-state-cb)) ;; コールバック関数の設定, #'で指定するのがEuslispのポイント

;; advertiseの設定
;;  どのtopicをpublishするか (指令値などの送信用)
(dotimes (i *arm-dof*)
  (ros::advertise (format nil "/arm_j~d_controller/command" (1+ i)) ;; topic名
                  std_msgs::Float64 ;; topicの型
                  1)) ;; キューの数


;; my subscribeの設定
;; my_cb_func
(defun is_ready_cb (msg)
  )
;;subscriber
(ros::subscribe "/isReady";; name of topoic you want to subscribe)
                std_msgs::Float64
                #'is_ready_cb)
;; advertise は上のやつでもう十分？？

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; サンプル関数
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defun test-read-state () ;; 各サーボの状態を全軸分読み出す
  (ros::spin-once)
  (pprint (send-all *state-list* :slots))
  t)
(warn ";; (test-read-state)~%")



(defun test-send-angle-vector-once () ;; 各軸、一回ずつ関節を動かす
  (ros::spin-once)
  ;; 現在関節角度からそれぞれ10度動かす
  (let ((current-av (mapcar #'rad2deg (send-all *state-list* :current_pos))))
    (dotimes (i *arm-dof*)
      (format t ";; send command position to arm ~d~%;; press enter~%" (1+ i))
      (read-line)
      (ros::publish (format nil "/arm_j~d_controller/command" (1+ i)) ;; topic名
                    (instance std_msgs::Float64 :init :data (deg2rad (+ 10 (elt current-av i))))) ;; 指令値をstd_msgs::Float64に変換して送信
      )
    t))
(warn ";; (test-send-angle-vector-once) ~%")



(defun test-send-angle-vector-loop (&key (move nil)) ;; ループで各関節を動かすサンプル
  (ros::spin-once)
  (let ((current-av '(0 0 0 0 0 0 0))
        (ii 0))
    (do-until-key
     (dotimes (i *arm-dof*)
       (format t ";; send for arm ~d~%;; press enter~%" (1+ i))
       (ros::publish (format nil "/arm_j~d_controller/command" (1+ i))
                     (if move
                         (instance std_msgs::Float64 :init :data (deg2rad (+ (* (- (mod ii 3) 1) 5) (elt current-av i))))
                       (instance std_msgs::Float64 :init :data (deg2rad (+ 0 (elt current-av i))))))
       )
     (format t ";; temps ~A~%" (send-all *state-list* :motor_temps))
     (unix:usleep 1000000) ;; 1秒まつ
     (incf ii)
     )
    t))
(warn ";; (test-send-angle-vector-loop :move nil) ~%")
(warn ";; (test-send-angle-vector-loop :move t) ~%")


;; 自主プロ
(setq hand1_1 (list 0 1 2)) ; 0が手首、１，２が指
(setq hand1_2 (list 3 4 5))
(setq hand2_1 (list 6 7 8))
(setq hand2_2 (list 9 10 11))
(setq short_interval 1000000)
(setq open_degree -45)
(setq close_degree 5)
(setq rotate_degree 90)
(setq rotate_degree_inv -90)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Basic module functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defun close_gripper (hand)
    ;; set all 5 deg.
    (dolist (i (cdr hand))
        (ros::publish (format nil "/arm_j~d_controller/command" (1+ i))
                      (instance std_msgs::Float64 :init :data (deg2rad close_degree))))
      ;;(unix:usleep short_interval)
      )

(defun open_gripper (hand)
    ;; set all 5 deg.
    (dolist (i (cdr hand))
        (ros::publish (format nil "/arm_j~d_controller/command" (1+ i))
                      (instance std_msgs::Float64 :init :data (deg2rad open_degree))))
      ;;(unix:usleep short_interval)
      )

;;手首を+90回す　指は開いたまま
(defun rotate_wrist (hand)
        (ros::publish (format nil "/arm_j~d_controller/command" (1+ (car hand))) ;; topic名
                      (instance std_msgs::Float64 :init :data (deg2rad rotate_degree_inv) ;; 指令値をstd_msgs::Float64に変換して送信
        ;;(unix:usleep short_interval)
    )))

;;手首を+90回す　指は開いたまま
(defun rotate_wrist_inv (hand)
        (ros::publish (format nil "/arm_j~d_controller/command" (1+ (car hand))) ;; topic名
                      (instance std_msgs::Float64 :init :data (deg2rad rotate_degree) ;; 指令値をstd_msgs::Float64に変換して送信
        ;;(unix:usleep short_interval)
    )))

(defun set_cube ()
    (dolist (i (list 0 3 6 9))
        (ros::publish (format nil "/arm_j~d_controller/command" (1+ i)) ;; topic名
                      (instance std_msgs::Float64 :init :data (deg2rad open_degree)
                  )))
    (read-line)
    (dolist (i (list 0 3 6 9))
        (ros::publish (format nil "/arm_j~d_controller/command" (1+ i)) ;; topic名
                      (instance std_msgs::Float64 :init :data (deg2rad close_degree)
                  )))
)

(defun reset-av ()
  (dotimes (i *arm-dof*)
      (ros::publish (format nil "/arm_j~d_controller/command" (1+ i)) ;; topic名
                    (instance std_msgs::Float64 :init :data (deg2rad open_degree)
                )))
  )

(defun convert_num (num)
  (if (eql (/ num 10) 1)
      ())
  (if (eql (/ num 10) 2)
      ())
  (if (eql (/ num 10) 3)
      ())
  (if (eql (/ num 10) 4)
      ())
  (if (eql (/ num 10) 5)
      ())
  (if (eql (/ num 10) 6)
      ())
)

(defun convert_lst (lst)
  (mapcar #'convert_num lst)
  )

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; utility func
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defun rotate_face (hand)
    ;; 指を開く
    (open_gripper hand)
    (unix:usleep short_interval)

    ;;手首を+90回す　指は開いたまま
    (rotate_wrist hand)
    (unix:usleep short_interval)

    ;;　指を閉じる
    (close_gripper hand)
    (unix:usleep short_interval)

    ;; 手首を−９０回転させてもとに戻す
    (rotate_wrist_inv hand)
    (unix:usleep short_interval)

    )

(defun rotate_face_inv (hand)
    ;; 指を開く
    (open_gripper hand)
    (unix:usleep short_interval)

    ;;手首を+90回す　指は開いたまま
    (rotate_wrist_inv hand)
    (unix:usleep short_interval)

    ;;　指を閉じる
    (close_gripper hand)
    (unix:usleep short_interval)

    ;; 手首を−９０回転させてもとに戻す
    (rotate_wrist hand)
    (unix:usleep short_interval)

  )

(defun rotate_cube1 ()
    ;;;;;;;;;;;;;;; Phse 1 ;;;;;;;;;;;;;;;
    ;;open hand2s gripper
    (open_gripper hand2_1)
    (open_gripper hand2_2)
    (unix:usleep short_interval)

    ;;rotate hand1s
    (rotate_wrist hand1_1)
    (rotate_wrist_inv hand1_2)
    (unix:usleep short_interval)

    ;;close hand2s gripper
    (close_gripper hand2_1)
    (close_gripper hand2_2)
    (unix:usleep short_interval)

    ;;横になってキューブを掴んでいる２つのハンドを、離して初期位置に戻す
    (open_gripper hand1_1)
    (open_gripper hand1_2)
    (unix:usleep short_interval)

    ;;reset hand1s (指はまだ開いたまま)
    (rotate_wrist_inv hand1_1)
    (rotate_wrist hand1_2)
    (unix:usleep short_interval)

    ;;ハンド１の指を閉じる
    (close_gripper hand1_1)
    (close_gripper hand1_2)
    (unix:usleep short_interval)

    ;;;;;;;;;;;;;;;;;;;;;;; Phase 2 ;;;;;;;;;;;;;;;;;

    ;;open hand1s gripper
    (open_gripper hand1_1)
    (open_gripper hand1_2)
    (unix:usleep short_interval)

    ;;rotate hand2s
    (rotate_wrist hand2_1)
    (rotate_wrist_inv hand2_2)
    (unix:usleep short_interval)

    ;;close hand1s gripper
    (close_gripper hand1_1)
    (close_gripper hand1_2)
    (unix:usleep short_interval)

    ;;横になってキューブを掴んでいる２つのハンドを、離して初期位置に戻す
    (open_gripper hand2_1)
    (open_gripper hand2_2)
    (unix:usleep short_interval)

    ;;reset hand2s (指はまだ開いたまま)
    (rotate_wrist_inv hand2_1)
    (rotate_wrist hand2_2)
    (unix:usleep short_interval)

    ;;ハンド2の指を閉じる
    (close_gripper hand2_1)
    (close_gripper hand2_2)
    (unix:usleep short_interval)
)

(defun rotate_cube2 ()
    ;;;;;;;;;;;;;;;;;;;;;;; Phase 1 ;;;;;;;;;;;;;;;;;

    ;;open hand1s gripper
    (open_gripper hand1_1)
    (open_gripper hand1_2)
    (unix:usleep short_interval)

    ;;rotate hand2s
    (rotate_wrist hand2_1)
    (rotate_wrist_inv hand2_2)
    (unix:usleep short_interval)

    ;;close hand1s gripper
    (close_gripper hand1_1)
    (close_gripper hand1_2)
    (unix:usleep short_interval)

    ;;横になってキューブを掴んでいる２つのハンドを、離して初期位置に戻す
    (open_gripper hand2_1)
    (open_gripper hand2_2)
    (unix:usleep short_interval)

    ;;reset hand2s (指はまだ開いたまま)
    (rotate_wrist_inv hand2_1)
    (rotate_wrist hand2_2)
    (unix:usleep short_interval)

    ;;ハンド2の指を閉じる
    (close_gripper hand2_1)
    (close_gripper hand2_2)
    (unix:usleep short_interval)

    ;;;;;;;;;;;;;;; Phse 2 ;;;;;;;;;;;;;;;
    ;;open hand2s gripper
    (open_gripper hand2_1)
    (open_gripper hand2_2)
    (unix:usleep short_interval)

    ;;rotate hand1s
    (rotate_wrist hand1_1)
    (rotate_wrist_inv hand1_2)
    (unix:usleep short_interval)

    ;;close hand2s gripper
    (close_gripper hand2_1)
    (close_gripper hand2_2)
    (unix:usleep short_interval)

    ;;横になってキューブを掴んでいる２つのハンドを、離して初期位置に戻す
    (open_gripper hand1_1)
    (open_gripper hand1_2)
    (unix:usleep short_interval)

    ;;reset hand1s (指はまだ開いたまま)
    (rotate_wrist_inv hand1_1)
    (rotate_wrist hand1_2)
    (unix:usleep short_interval)

    ;;ハンド１の指を閉じる
    (close_gripper hand1_1)
    (close_gripper hand1_2)
    (unix:usleep short_interval)
)

(defun convert_num (num)
  (if (eql (/ num 10) 1)
      ())
  (if (eql (/ num 10) 2)
      ())
  (if (eql (/ num 10) 3)
      ())
  (if (eql (/ num 10) 4)
      ())
  (if (eql (/ num 10) 5)
      ())
  (if (eql (/ num 10) 6)
      ())
)

(defun convert_lst (lst)
  (mapcar #'convert_num lst)
  )

(defun solve_cube (lst)
  (let ((x (car lst)))
      (if (null x)
          (set_cube);;終了操作
          (progn
          ;; if 1, rotate_face
              (if (eql (/ x 10) 1)
                  (progn (rotate_face hand1_1)
                         (solve_cube (cdr lst))))
              ;; 2
              (if (eql (/ x 10) 2)
                  (progn (rotate_face hand1_1)
                          (solve_cube (cdr lst))))
              ;; 3
              (if (eql (/ x 10) 3)
                  (progn (rotate_face hand1_1)
                          (solve_cube (cdr lst))))
              ;; 4
              (if (eql (/ x 10) 4)
                  (progn (rotate_face hand1_1)
                        (solve_cube (cdr lst))))
              ;; if 5, rotate-cube
              (if (eql (/ x 10) 5)
                  (progn (rotate_cube1)
                          (solve_cube (convert_num lst)))
              )
              ;; if 6, rotate-cube
              (if (eql (/ x 10) 6)
                  (progn (rotate_cube1)
                          (solve_cube (convert_num lst)))
              )
           )
  )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;     main program
;;
;; topic; 'plz_rotate' Float64 camera --> robot
;;        'solution' Num camera --> robot
;;        'isReady' Float64 robot --> camera
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(set_cube)
;; camera --> robot on "plx_rotate" [スキャンしたからキューブをまわせ]

;;実際にまわしてスキャンする準備をする
;;robot --> camera on "isReady" [準備したそ、スキャンしろ]
;;　２回目　camera --> robot on "plx_rotate" [スキャンしたからキューブをまわせ]
;;　２回目　実際にまわしてスキャンする準備をする
;;  ２回目　robot --> camera on "isReady" [準備したそ、スキャンしろ]
;;　camera --> robot on "solution" [３面スキャン終わった、答え送ったからこれどおりに溶け]
;;
