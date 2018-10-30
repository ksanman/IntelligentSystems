#|
===================================
-*- Mode: Lisp; Syntax: Common-Lisp -*-

File: ca-words.lisp
Author: Vladimir Kulyukin
Description: Simple word definitions.

===================================
|#

(in-package "USER")

;;; (gethash 'jack *ca-lexicon*) returns
;;; ((concept nil (human :name (jack) :sex (male)))).
(define-ca-word 
    jack
    (concept nil (human :name (jack) :sex (male))))

(define-ca-word
    john
    (concept nil (human :name (john) :sex (male))))

(define-ca-word
    ann
    (concept nil (human :name (ann) :sex (female))))

(define-ca-word
    logan
    (concept nil (city :name (logan))))

(define-ca-word
    preston
    (concept nil (city :name (preston))))

(define-ca-word
    book
    (concept nil (book)))

;;; the effect of defining jack is as follows:
;;; (gethash 'jack *ca-lexicon*) returns
;;; ((WORDS -PAST- -EAT-)).
(define-ca-word
    ate
    (words -past- -eat-))

(define-ca-word
    went
    (words -past- -go-))

(define-ca-word
    gave
    (words -past- -give-))

(define-ca-word
    to
  (concept ?dir-to (direction-to))
  (concept ?reception-to (reception-to))
  (request (test (after ?dir-to ?loc (location)))
	   (actions (modify ?dir-to :loc ?loc)))
  (request (test (after ?reception-to ?target (human)))
	   (actions (modify ?dir-to :target ?target))))

(define-ca-word
    from
    (concept ?dir-from (direction-from))
  (request (test (after ?dir-from ?loc (location)))
	   (actions (modify ?dir-from :loc ?loc))))

;;; (define-ca-word
;;;    eats
;;;    (words -present- -eat-))

;;; (define-ca-word
;;;    is
;;;    (words -be-present- -be-))

;;; (define-ca-word
;;;    eating
;;;    (words -gerund- -eat-))

;(define-ca-word 
;    -eat-
;    (concept ?act (ingest))
;  (if (before nil ?x (animate))
;      (modify ?act :actor ?x))
;  (request (test (after ?act ?food (food)))
;	   (actions (modify ?act :object ?food))))  

(define-ca-word
    -eat-
    (concept ?act (ingest))
  (request (test (before ?act ?actor (animate)))
	   (actions (modify ?act :actor ?actor)))
  (request (test (after ?act ?food (food)))
	   (actions (modify ?act :object ?food))))

(define-ca-word
    -go-
    (concept ?act (ptrans))
  (request (test (before ?act ?actor (animate)))
	   (actions (modify ?act :actor ?actor)))
  (request (test (after ?act ?from-loc (direction-from)))
	   (actions (modify ?act :from ?from-loc)))
  (request (test (after ?act ?to-loc (direction-to)))
	   (actions (modify ?act :to ?to-loc))))

(define-ca-word
    -give-
    (concept ?act (atrans))
  (request (test (before ?act ?actor (human)))
	   (actions (modify ?act :actor ?actor)))
  (request (test (after ?act ?reception-to (reception-to)))
	   (actions (modify ?act :to ?reception-to)))
  (request (test (after ?act ?recipient (human)))
	   (actions (modify ?act :to ?recipient)))
  (request (test (after ?act ?object (inanimate)))
	   (actions (modify ?act :object ?object))))

(define-ca-word 
    -past-
    (mark ?x)
  (request (test (after ?x ?act (action)))
	   (actions (modify ?act :time (past)))))

;;; (define-ca-word
;;;    -be-
;;;    (concept ?copula (be)))

;;;(define-ca-word
;;;    -present-
;;;    (mark ?x)
;;;  (request (test (after ?x ?act (action)))
;;;	   (actions (modify ?act :time (present)))))

;;;(define-ca-word
;;;    -be-present-
;;;    (mark ?x)
;;;  (request (test (after ?x ?copula (be)))
;;;	   (actions (modify ?copula :time (present)))))

;;;(define-ca-word
;;;    -gerund-
;;;    (mark ?x)
;;;  (request (test (and (before nil ?copula (be :time (present)))
;;;		      (after ?x ?act (action)))
;;;	   (actions (modify ?act :time (present-cont))))))
  
(define-ca-word 
    an
    (mark ?x)
  (request (test (after ?x ?con (concept)))
	   (actions (modify ?con :ref (indef)))))

(define-ca-word 
    apple
    (concept nil (apple)))

(define-ca-word
    pear
    (concept nil (pear)))

;;; end-of-file

