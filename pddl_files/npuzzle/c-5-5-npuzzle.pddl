(define (problem uzzle)
  (:domain cpuzzle)
  (:objects
    ;; Tiles
    tile-1 tile-2 tile-3 tile-4 tile-5 - tile
    tile-6 tile-7 tile-8 tile-9 tile-10 - tile
    tile-11 tile-12 tile-13 tile-14 tile-15 - tile
    tile-16 tile-17 tile-18 tile-19 tile-20 - tile
    tile-21 tile-22 tile-23 tile-24 - tile

    ;; Cells
    cell-1-1 cell-1-2 cell-1-3 cell-1-4 cell-1-5 - cell
    cell-2-1 cell-2-2 cell-2-3 cell-2-4 cell-2-5 - cell
    cell-3-1 cell-3-2 cell-3-3 cell-3-4 cell-3-5 - cell
    cell-4-1 cell-4-2 cell-4-3 cell-4-4 cell-4-5 - cell
    cell-5-1 cell-5-2 cell-5-3 cell-5-4 cell-5-5 - cell
  )

  (:init
    ;; Initial tile positions
    (at tile-1 cell-1-1)
    (at tile-2 cell-1-2)
    (at tile-3 cell-1-3)
    (at tile-4 cell-1-4)
    (at tile-5 cell-1-5)
    (at tile-6 cell-2-1)
    (at tile-7 cell-2-2)
    (at tile-8 cell-2-3)
    (at tile-9 cell-2-4)
    (at tile-10 cell-2-5)
    (at tile-11 cell-3-1)
    (at tile-12 cell-3-2)
    (at tile-13 cell-3-3)
    (at tile-14 cell-3-4)
    (at tile-15 cell-3-5)
    (at tile-16 cell-4-1)
    (at tile-17 cell-4-2)
    (at tile-18 cell-4-3)
    (at tile-19 cell-4-4)
    (at tile-20 cell-4-5)
    (at tile-21 cell-5-1)
    (at tile-22 cell-5-2)
    (at tile-23 cell-5-3)
    (at tile-24 cell-5-4)

    ;; Blank position
    (blank cell-5-5)

    ;; Adjacency relations (full)
    ;; Horizontal (right and left)
    (right cell-1-1 cell-1-2)
    (right cell-1-2 cell-1-3)
    (right cell-1-3 cell-1-4)
    (right cell-1-4 cell-1-5)

    (right cell-2-1 cell-2-2)
    (right cell-2-2 cell-2-3)
    (right cell-2-3 cell-2-4)
    (right cell-2-4 cell-2-5)

    (right cell-3-1 cell-3-2)
    (right cell-3-2 cell-3-3)
    (right cell-3-3 cell-3-4)
    (right cell-3-4 cell-3-5)

    (right cell-4-1 cell-4-2)
    (right cell-4-2 cell-4-3)
    (right cell-4-3 cell-4-4)
    (right cell-4-4 cell-4-5)

    (right cell-5-1 cell-5-2)
    (right cell-5-2 cell-5-3)
    (right cell-5-3 cell-5-4)
    (right cell-5-4 cell-5-5)

    ;; Vertical (down and up)
     (up cell-2-1 cell-1-1)
     (up cell-2-2 cell-1-2)
     (up cell-2-3 cell-1-3)
     (up cell-2-4 cell-1-4)
     (up cell-2-5 cell-1-5)

     (up cell-3-1 cell-2-1)
     (up cell-3-2 cell-2-2)
     (up cell-3-3 cell-2-3)
     (up cell-3-4 cell-2-4)
     (up cell-3-5 cell-2-5)

     (up cell-4-1 cell-3-1)
     (up cell-4-2 cell-3-2)
     (up cell-4-3 cell-3-3)
     (up cell-4-4 cell-3-4)
     (up cell-4-5 cell-3-5)

     (up cell-5-1 cell-4-1)
     (up cell-5-2 cell-4-2)
     (up cell-5-3 cell-4-3)
    (up cell-5-4 cell-4-4)
    (up cell-5-5 cell-4-5)
  )

  (:goal
    (and
      (at tile-1 cell-1-1)
      (at tile-2 cell-1-2)
      (at tile-3 cell-1-3)
      (at tile-4 cell-1-4)
      (at tile-5 cell-1-5)
      (at tile-6 cell-2-1)
      (at tile-7 cell-2-2)
      (at tile-8 cell-2-3)
      (at tile-9 cell-2-4)
      (at tile-10 cell-2-5)
      (at tile-11 cell-3-1)
      (at tile-12 cell-3-2)
      (at tile-13 cell-3-3)
      (at tile-14 cell-3-4)
      (at tile-15 cell-3-5)
      (at tile-16 cell-4-1)
      (at tile-17 cell-4-2)
      (at tile-18 cell-4-3)
      (at tile-19 cell-4-4)
      (at tile-20 cell-4-5)
      (at tile-21 cell-5-1)
      (at tile-22 cell-5-2)
      (at tile-23 cell-5-3)
      (at tile-24 cell-5-4)
      (blank cell-5-5)
    )
  )
)