;;; -*- Mode: Lisp; Syntax: Common-Lisp; -*-
;;;; Module: gps2.lisp: 2nd version of GPS
;;; =========================================

(in-package :user)

(require :auxfuns "auxfuns")

(defstruct op "An operation"
  (action nil) 
  (preconds nil) 
  (add-list nil) 
  (del-list nil))

(defun executing-p (x)
  "Is x of the form: (execute ...) ?"
  (starts-with x 'execute))

(defun convert-op (op)
  "Make op conform to the (EXECUTING op) convention."
  (unless (some #'executing-p (op-add-list op))
    (push (list 'execute (op-action op)) (op-add-list op)))
  op)

(defun op (action &key preconds add-list del-list)
  "Make a new operator that obeys the (EXECUTING op) convention."
  (convert-op
    (make-op :action action :preconds preconds
             :add-list add-list :del-list del-list)))

;;; ==============================

(defparameter *school-ops*
  (list
    (make-op :action 'drive-son-to-school
         :preconds '(son-at-home car-works)
         :add-list '(son-at-school)
         :del-list '(son-at-home))
    (make-op :action 'shop-installs-battery
         :preconds '(car-needs-battery shop-knows-problem shop-has-money)
         :add-list '(car-works))
    (make-op :action 'tell-shop-problem
         :preconds '(in-communication-with-shop)
         :add-list '(shop-knows-problem))
    (make-op :action 'telephone-shop
         :preconds '(know-phone-number)
         :add-list '(in-communication-with-shop))
    (make-op :action 'look-up-number
         :preconds '(have-phone-book)
         :add-list '(know-phone-number))
    (make-op :action 'give-shop-money
         :preconds '(have-money)
         :add-list '(shop-has-money)
         :del-list '(have-money))))

(mapc #'convert-op *school-ops*)

;;; ==============================

(defvar *ops* nil "A list of available operators.")

(defstruct op "An operation"
  (action nil) (preconds nil) (add-list nil) (del-list nil))

(defun GPS (state goals &optional (*ops* *ops*))
  "General Problem Solver: from state, achieve goals using *ops*."
  (remove-if #'atom (achieve-all (cons '(start) state) goals nil)))

;;; ==============================

(defun achieve-all (state goals goal-stack)
  "Achieve each goal, and make sure they still hold at the end."
  (let ((current-state state))
    (if (and (every #'(lambda (g)
                        (setf current-state
                              (achieve current-state g goal-stack)))
                    goals)
             (subsetp goals current-state :test #'equal))
        current-state)))

(defun achieve (state goal goal-stack)
  "A goal is achieved if it already holds,
  or if there is an appropriate op for it that is applicable."
  (dbg-indent :gps (length goal-stack) "Goal: ~a" goal)
  (cond ((member-equal goal state) state)
        ((member-equal goal goal-stack) nil)
        (t (some #'(lambda (op) (apply-op state goal op goal-stack))
                 (find-all goal *ops* :test #'appropriate-p)))))

;;; ==============================

(defun member-equal (item list)
  (member item list :test #'equal))

;;; ==============================

(defun apply-op (state goal op goal-stack)
  "Return a new, transformed state if op is applicable."
  (dbg-indent :gps (length goal-stack) "Consider: ~a" (op-action op))
  (let ((state2 (achieve-all state (op-preconds op) 
                             (cons goal goal-stack))))
    (unless (null state2)
      ;; Return an updated state
      (dbg-indent :gps (length goal-stack) "Action: ~a" (op-action op))
      (append (remove-if #'(lambda (x) 
                             (member-equal x (op-del-list op)))
                         state2)
              (op-add-list op)))))

(defun appropriate-p (goal op)
  "An op is appropriate to a goal if it is in its add list."
  (member-equal goal (op-add-list op)))

;;; ==============================

(defun use (oplist)
  "Use oplist as the default list of operators."
  ;; Return something useful, but not too verbose: 
  ;; the number of operators.
  (length (setf *ops* oplist)))

;;; ==============================

(defparameter *banana-ops*
  ;;; REPLACE NIL WITH YOUR LIST OF OPERATORS.
   (list
       (make-op :action 'eat-bananas
         :preconds '(has-bananas hungry)
         :add-list '(not-hungry)
         :del-list '(has-bananas hungry))
      (make-op :action 'grasp-bananas
         :preconds '(empty-handed at-bananas)
         :add-list '(has-bananas)
         :del-list '(empy-handed at-bananas))
      (make-op :action 'drop-ball
         :preconds '(has-ball)
         :add-list '(empty-handed)
         :del-list '(has-ball))
      (make-op :action 'climb-on-chair
         :preconds '(at-chair on-floor chair-in-middle-of-room on-floor)
         :add-list '(at-bananas)
         :del-list '(on-floor at-chair))
      (make-op :action 'push-chair-to-middle-of-room
         :preconds '(chair-at-door at-door)
         :add-list '(chair-in-middle-of-room at-chair)
         :del-list '(chair-at-door at-door))))

(mapc #'convert-op *banana-ops*)

;;; ==============================

(defvar *ops* nil "A list of available operators.")

(defstruct op "An operation"
  (action nil) (preconds nil) (add-list nil) (del-list nil))

(defun GPS (state goals &optional (*ops* *ops*))
  "General Problem Solver: from state, achieve goals using *ops*."
  (remove-if #'atom (achieve-all (cons '(start) state) goals nil)))

;;; ==============================

(defun achieve-all (state goals goal-stack)
  "Achieve each goal, and make sure they still hold at the end."
  (let ((current-state state))
    (if (and (every #'(lambda (g)
                        (setf current-state
                              (achieve current-state g goal-stack)))
                    goals)
             (subsetp goals current-state :test #'equal))
        current-state)))

(defun achieve (state goal goal-stack)
  "A goal is achieved if it already holds,
  or if there is an appropriate op for it that is applicable."
  (dbg-indent :gps (length goal-stack) "Goal: ~a" goal)
  (cond ((member-equal goal state) state)
        ((member-equal goal goal-stack) nil)
        (t (some #'(lambda (op) (apply-op state goal op goal-stack))
                 (find-all goal *ops* :test #'appropriate-p)))))

;;; ==============================

(defun member-equal (item list)
  (member item list :test #'equal))

;;; ==============================

(defun apply-op (state goal op goal-stack)
  "Return a new, transformed state if op is applicable."
  (dbg-indent :gps (length goal-stack) "Consider: ~a" (op-action op))
  (let ((state2 (achieve-all state (op-preconds op) 
                             (cons goal goal-stack))))
    (unless (null state2)
      ;; Return an updated state
      (dbg-indent :gps (length goal-stack) "Action: ~a" (op-action op))
      (append (remove-if #'(lambda (x) 
                             (member-equal x (op-del-list op)))
                         state2)
              (op-add-list op)))))

(defun appropriate-p (goal op)
  "An op is appropriate to a goal if it is in its add list."
  (member-equal goal (op-add-list op)))

;;; ==============================

(defun use (oplist)
  "Use oplist as the default list of operators."
  ;; Return something useful, but not too verbose: 
  ;; the number of operators.
  (length (setf *ops* oplist)))

;;; ==============================

;;;(defparameter *banana-ops*
;;;  (list
;;;    (op 'climb-on-chair
;;;        :preconds '(chair-at-middle-room at-middle-room on-floor)
;;;        :add-list '(at-bananas on-chair)
;;;        :del-list '(at-middle-room on-floor))
;;;    (op 'push-chair-from-door-to-middle-room
;;;        :preconds '(chair-at-door at-door)
;;;        :add-list '(chair-at-middle-room at-middle-room)
;;;        :del-list '(chair-at-door at-door))
;;;    (op 'walk-from-door-to-middle-room
;;;        :preconds '(at-door on-floor)
;;;        :add-list '(at-middle-room)
;;;        :del-list '(at-door))
;;;    (op 'grasp-bananas
;;;        :preconds '(at-bananas empty-handed)
;;;        :add-list '(has-bananas)
;;;        :del-list '(empty-handed))
;;;    (op 'drop-ball
;;;        :preconds '(has-ball)
;;;        :add-list '(empty-handed)
;;;        :del-list '(has-ball))
;;;    (op 'eat-bananas
;;;        :preconds '(has-bananas)
;;;        :add-list '(empty-handed not-hungry)
;;;        :del-list '(has-bananas hungry))))

;;; ==============================

(defun gps (state goals &optional (*ops* *ops*))
  "General Problem Solver: from state, achieve goals using *ops*."
  (find-all-if #'action-p
               (achieve-all (cons '(start) state) goals nil)))

(defun action-p (x)
  "Is x something that is (start) or (executing ...)?"
  (or (equal x '(start)) (executing-p x)))

;;; ==============================

(defun find-path (start end)
  "Search a maze for a path from start to end."
  (let ((results (GPS `((at ,start)) `((at ,end)))))
    (unless (null results)
      (cons start (mapcar #'destination
                          (remove '(start) results
                                  :test #'equal))))))

(defun destination (action)
  "Find the Y in (executing (move from X to Y))"
  (fifth (second action)))

;;; ==============================

(defun achieve-all (state goals goal-stack)
  "Achieve each goal, trying several orderings."
  (some #'(lambda (goals) (achieve-each state goals goal-stack))
        (orderings goals)))

(defun achieve-each (state goals goal-stack)
  "Achieve each goal, and make sure they still hold at the end."
  (let ((current-state state))
    (if (and (every #'(lambda (g)
                        (setf current-state
                              (achieve current-state g goal-stack)))
                    goals)
             (subsetp goals current-state :test #'equal))
        current-state)))

(defun orderings (l) 
  (if (> (length l) 1)
      (list l (reverse l))
      (list l)))

;;; ==============================

(defun achieve (state goal goal-stack)
  "A goal is achieved if it already holds,
  or if there is an appropriate op for it that is applicable."
  (dbg-indent :gps (length goal-stack) "Goal: ~a" goal)
  (cond ((member-equal goal state) state)
        ((member-equal goal goal-stack) nil)
        (t (some #'(lambda (op) (apply-op state goal op goal-stack))
                 (appropriate-ops goal state))))) ;***

(defun appropriate-ops (goal state)
  "Return a list of appropriate operators, 
  sorted by the number of unfulfilled preconditions."
  (sort (copy-list (find-all goal *ops* :test #'appropriate-p)) #'<
        :key #'(lambda (op) 
                 (count-if #'(lambda (precond)
                               (not (member-equal precond state)))
                           (op-preconds op)))))

;;; ==============================

(defun trace-gps ()
  (do-debug :gps))

(defun untrace-gps ()
  (do-undebug))

