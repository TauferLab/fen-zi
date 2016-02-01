!     Program to analyze a CHARMM trajectory file
!     Joe Davis, August 2008

!     Usage:
!     trjanalysis < infile outfile

!     Sample input file format:
!     -20.0             minimum for histogram
!     20.0              maximum for histogram
!     1000              number of histogram bins
!     256               number of atoms
!     2                 number of trajectory files
!     false             flag for charges; true=read from trj, false=read from charge.dat
!     false             flag for masses; true=read from mass.dat, false=do not read
!     dyn1.trj          list of trj files
!     dyn2.trj

!     Final histogram will be written to outfile, errors and progress will be written
!     to standard output

!     Under "analysis code goes here", there is a simple number density calculation.
!     You can put whatever analysis and binning you want there. All coordinates
!     (and charges/masses if desired) will have been read. The calculation/binning will
!     be done for each frame in the trajectory. If you want a time profile, put a write
!     statement to unit 10 (output file) in with the analysis and skip the histogram part.

      program analysis

      implicit none

      integer ICNTRL(20), NTITLE
      integer NATREC, ICONF, IUNIT
      integer I, J, count, trjnum, narg, nframe, natom
      integer nbins, ibin
      real zlow, zhi, zpos, delta, LX, LY, LZ
      real(8) XTLABC(6)
      real(8), allocatable, dimension(:) :: X, Y, Z, CG, mass
      real, allocatable, dimension(:) :: TEMP, hist
      integer, allocatable, dimension(:) :: ncount
      character(4) HDRR
      character(80) TITLE(32) 
      character(200) trjfile
      character(200) outfile
      character(8) fnum
      logical QCRYS, QDIM4, QHAS_CHGS, mass_read

      real comx, comy, comz, mtot, comx2, comy2, comz2, mtot2
      real gacomx, gacomy, gacomz, zfind

!--------------------------------------------------------
!     Read histogram parameters
!--------------------------------------------------------
      read(5,*) zlow
      read(5,*) zhi
      read(5,*) nbins
      delta = (zhi-zlow)/(real(nbins))

!     Dynamically allocate histogram arrays indexed from 0 to nbins
      allocate(hist(0:nbins), ncount(0:nbins))

!     Zero out histogram elements
      do I = 0,nbins
         hist(I) = 0.0
         ncount(I) = 0
      enddo

!--------------------------------------------------------
!     Read input parameters
!--------------------------------------------------------

      IUNIT = 55

!     Get number of command line arguments
      narg = iargc()

!     Read the number of atoms from standard input
      read(5,*) natom
      write(*,*) natom, 'atoms expected'

!     Dynamically allocate arrays of size natom
      allocate(X(natom), Y(natom), Z(natom), CG(natom), mass(natom),
     $         TEMP(natom))

!     Read the number of trj files
      read(5,*) trjnum
      write(*,*) trjnum, ' trj file(s) will be read'

!     Read flag for charges
      read(5,*) QHAS_CHGS

      if(QHAS_CHGS) then
         write(*,*) 'Charges will be read from trajectory'
      else
         write(*,*) 'Charges will be read from charge.dat'
      endif

!     Read flag for masses
      read(5,*) mass_read

      if(mass_read) then
         write(*,*) 'Masses will be read from mass.dat'
      else
         write(*,*) 'Masses will not be read'
      endif

!     Read output file, default to 'data.dat' if none specified
      if(narg .eq. 0) then
         outfile = 'find.dat'
      else
         call getarg(1, outfile)
      endif

      write(*,*) 'Output file: ', trim(outfile)
      write(*,*)

!     Read masses
      if(mass_read) then
         open(unit=99,file='mass.dat',status='old',err=10)
         rewind(99)
         DO I = 1,natom
            read(99,*) mass(I)
         ENDDO
         close(99)
      endif

      goto 20

 10   write(*,*) 'Error opening mass.dat, masses will not be read'
      write(*,*)

!     Read charges (if not in trj file)
 20   if(.not. QHAS_CHGS) then
         open(unit=99,file='charge.dat',status='old',err=30)
         rewind(99)
         DO I = 1,natom
            read(99,*) CG(I)
         ENDDO
         close(99)
      endif

      goto 40

 30   write(*,*) 'Error opening charge.dat, charges will not be read'
      write(*,*)

 40   nframe = 0

!	Open output file
      open(unit=10,file=trim(outfile),status='replace',FORM='FORMATTED')


!--------------------------------------------------------
!     Read CHARMM trajectory files
!--------------------------------------------------------

      do count = 1, trjnum      ! Loop over trj files

!     Read list of trj files from standard input
         read(5,'(A200)') trjfile

         write(*,*) 'Reading ', trim(trjfile)

!     Open trj file
         open(unit=IUNIT,file=trim(trjfile), status='old',
     $        FORM='UNFORMATTED')

         rewind(IUNIT)

!     Read title, number of atoms, etc.
!     For every read, if end of file is reached, branch to end of the loop and
!     move on to the next trj file
         READ(IUNIT,end=100) HDRR, ICNTRL
         ! READ(IUNIT,end=100) NTITLE,(TITLE(I),I=1,NTITLE)
         READ(IUNIT,end=100) NTITLE, (TITLE(I),I=1,NTITLE)
         WRITE(*,*) HDRR, ICNTRL, NTITLE, TITLE(1)
         STOP
         READ(IUNIT,end=100) NATREC

!     Exit if trj file does not have the expected number of atoms
         if(NATREC .ne. natom) then
            write(*,*)
            write(*,*) 'Error: ', trim(trjfile), ' has'
            write(*,*) NATREC, 'atoms;', natom, 'expected'
            close(IUNIT)
            stop
         endif

         QDIM4=(ICNTRL(12).EQ.1)

         QCRYS = .true.

         DO ICONF = 1,ICNTRL(1) ! Loop over frames in trajectory

!     Read dimensions, coordinates, charges

            IF(QCRYS) then
               READ(IUNIT,end=100) XTLABC
            endif
            READ(IUNIT,end=100) TEMP
            DO I=1,NATOM
               X(I)=TEMP(I)
            ENDDO
            READ(IUNIT,end=100) TEMP
            DO I=1,NATOM
               Y(I)=TEMP(I)
            ENDDO
            READ(IUNIT,end=100) TEMP
            DO I=1,NATOM
               Z(I)=TEMP(I)
            ENDDO
            IF(QDIM4) THEN
               READ(IUNIT,end=100) TEMP
            ENDIF
            IF (QHAS_CHGS) THEN
               READ(IUNIT,end=100) TEMP
               DO I=1,NATOM
                  CG(I)=TEMP(I)
               ENDDO
            ENDIF

!     Get box dimensions
            LX = XTLABC(1)
            LY = XTLABC(3)
            LZ = XTLABC(6)

!     At this point, the coordinates (and charges if not fixed charge) have been read
!     for this frame in the trajectory

!--------------------------------------------------------
!     Analysis code starts here
!--------------------------------------------------------

!     loop over lipid to calculate center of mass

            comx = 0.0
            comy = 0.0
            comz = 0.0
            mtot = 0.0

            do I = 1, 9360
               comx = comx + X(I)*mass(I)
               comy = comy + Y(I)*mass(I)
               comz = comz + Z(I)*mass(I)
               mtot = mtot + mass(I)
            enddo

            comx = comx / mtot
            comy = comy / mtot
            comz = comz / mtot

!     loop over mguan to calculate center of mass

            comx2 = 0.0
            comy2 = 0.0
            comz2 = 0.0
            mtot2 = 0.0

            do I = 22173, 22185
               comx2 = comx2 + X(I)*mass(I)
               comy2 = comy2 + Y(I)*mass(I)
               comz2 = comz2 + Z(I)*mass(I)
               mtot2 = mtot2 + mass(I)
            enddo

            comx2 = comx2 / mtot2
            comy2 = comy2 / mtot2
            comz2 = comz2 / mtot2

!     find next window
            zfind = comz2-comz
                write(*,*) ICONF, zfind

!--------------------------------------------------------
!     Analysis code ends here
!--------------------------------------------------------

         ENDDO                  ! Loop over frames in trajectory (ICONF)

 100     close(IUNIT)

!         write(*,*) ICONF-1, 'frames read'
!         write(*,*)
         nframe = nframe + ICONF - 1

      enddo                     ! Loop over trj files (count)

!      write(*,*) nframe, 'total frames read'

!--------------------------------------------------------
!     Output histogram
!--------------------------------------------------------

!!     Write out histogram
!      open(unit=10,file=trim(outfile),status='replace',FORM='FORMATTED')

!      do I=1,nbins
!         zpos = zlow + (I-1)*delta 
!         write(10,*) zpos,hist(I)/real(2*nframe*72)
!      enddo

!     Deallocate arrays
      deallocate(X, Y, Z, CG, TEMP, hist, ncount)

      end
