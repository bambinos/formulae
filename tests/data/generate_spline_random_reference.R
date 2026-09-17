# Run from the repository root. Generates fixtures. R is not required by pytest.
# Reference: mgcv::smoothCon(scale.penalty=FALSE) and smooth2random(type=2).
library(mgcv)
library(jsonlite)

x <- c(0, .2, .2, .8, 1.5, 2, 2.7, 3.5, 4.2, 5.1, 6.4, 7.3, 8.1, 9.2, 10)
new_x <- c(-2, 0, .4, 2.5, 8, 10, 12, 22)
cases <- list()

for (kind in c("cr", "cc", "tp")) {
    for (center in c(FALSE, TRUE)) {
        knots <- switch(kind, cr=c(0, 2, 5, 8, 10), cc=c(0, 1, 3, 5, 8, 10), tp=NULL)
        spec <- s(x, bs=kind, k=if (kind == "cc") 6 else 5)
        sm <- smoothCon(
            spec,
            data.frame(x=x),
            knots=if (is.null(knots)) NULL else list(x=knots),
            absorb.cons=center,
            scale.penalty=FALSE
        )[[1]]

        re <- smooth2random(sm, "", type=2)
        prediction <- PredictMat(sm, data.frame(x=new_x))
        rotation <- sweep(re$trans.U, 2, re$trans.D, "*")
        converted <- prediction %*% rotation
        key <- paste(kind, tolower(center), sep="_")

        cases[[key]] <- list(
            X=sm$X,
            S=sm$S[[1]],
            Xnew=prediction,
            Z=cbind(re$Xf, re$rand[[1]]),
            Znew=cbind(
                converted[, re$pen.ind == 0, drop=FALSE],
                converted[, re$pen.ind != 0, drop=FALSE]
            )
        )
    }
}
write_json(
    list(
        mgcv_version=as.character(packageVersion("mgcv")),
        x=x,
        new_x=new_x,
        cases=cases
    ),
    "tests/data/spline_random_reference.json",
    digits=16,
    pretty=TRUE,
    auto_unbox=TRUE,
    matrix="rowmajor"
)
