require 'gnuplot'

class EMAlgorithm
  attr_accessor :dataset
  attr_accessor :gauss_mix_num
  attr_accessor :dataset_length
  attr_accessor :dimensions
  attr_accessor :params
  attr_accessor :clusters
  attr_accessor :reponsibilities

  #########ultilities
  #factorial
  #exponential matrix...
  ###################
  def factorial n
    ret = 1
    for i in 1..n
      ret = ret*i
    end
    return ret
  end

  def exponential_matrix input
    max = 200
    ret = Matrix.identity(input.column_size)
    max.times do |i|
      ret = ret + input**i / factorial(i)
    end
    return ret
  end
  
  #note that mean is Array and input is Matrix
  def substract_by_mean(input, mean)
    rows = []
    mean.each_with_index { |elem, i|
      rows[i] = []
      input.column_size.times {
        rows[i] << elem  
      }
    }    
    mean_mtx = Matrix.rows(rows, false)    
    return input - mean_mtx 
  end  
 
  def make_matrix input
    rows = []
    input.each {|elem|
      elem.each_with_index {|el, i|
        if rows[i] == nil
          rows[i] = []
        else
          rows[i] << el
        end
      }
    }
    return Matrix.rows(rows, false)
  end

  def matrix_to_float input
    input.each {|elem|
      return elem
    }
  end 
  ####################

  
  
  ##todo: refactoring kmeans to universal
  def kmeans input 
    ##initialize
    cluster1 = []
    cluster2 = []
    
    mean1 = [500, 500]
    mean2 = [850, 850]
  
    threshold = 1
    while threshold > 0 do
      cluster1 = []
      cluster2 = []
      input.each do |element|
        #puts (element[0].to_s + ' ' + element[1].to_s)
        dis_to_cluster1 = (element[0]-mean1[0])**2 + (element[1]-mean1[1])**2
        dis_to_cluster2 = (element[0]-mean2[0])**2 + (element[1]-mean2[1])**2
        if dis_to_cluster1 <= dis_to_cluster2
          cluster1 << element
        else
          cluster2 << element
        end
      end      
       
      ##recalculate mean
      sum1 = [0,0]
      cluster1.each {|element| 
        sum1[0] = sum1[0] + element[0]
        sum1[1] = sum1[1] + element[1]
      }
      new_mean1 = [sum1[0]/cluster1.length, sum1[1]/cluster1.length]
       
      sum2 = [0,0]
      cluster2.each {|element| 
        sum2[0] = sum2[0] + element[0]
        sum2[1] = sum2[1] + element[1]
      }
      new_mean2 = [sum2[0]/cluster2.length, sum2[1]/cluster2.length]

      #if diff bw new mean and old mean ==0 then converged   
      diff = (new_mean1[0]-mean1[0])**2 + (new_mean2[0] - mean2[0])**2 + (new_mean1[1]-mean1[1])**2 + (new_mean2[1] - mean2[1])**2
      if diff == 0
        #puts 'end' + cluster1.length.to_s + ' ' + cluster2.length.to_s
        ret_hash = {:mean => [], :clusters => []}
        ret_hash[:mean] << mean1 << mean2
        ret_hash[:clusters] << cluster1 << cluster2
        return ret_hash
      else
        mean1 = new_mean1
        mean2 = new_mean2
        ##puts 'update' + mean1[0].to_s + ' ' + mean1[1].to_s + '+++' + mean2[0].to_s + ' ' + mean2[1].to_s
      end
    end
  end

  ##todo: make covariance function universal
  #note: current function is just for 2xn dataset
  #dataset is Matrix of 2xn 
  def covariance dataset
    ##calculate mean of each dimension
    mean = []
    sum1, sum2 = 0, 0
    dataset.each_with_index { |elem, row, col| 
      if row == 0
        sum1 = sum1 + elem
      elsif row == 1
        sum2 = sum2 + elem
      end
    }  
    mean << sum1/dataset.column_size << sum2/dataset.column_size
  
    ##calculate covariance
    #make mean matrix
    row1, row2  = [], []
    dataset.column_size.times { 
      row1 << mean[0]
      row2 << mean[1]
    } 
    mean_mtx = Matrix[row1, row2]
    matx1 = dataset - mean_mtx 
    size = 1.0/(dataset.column_size - 1)
    puts ((matx1 * (matx1.transpose))*size).determinant()
    return (matx1 * (matx1.transpose))*size
  end

  #calculate expectation of a value with gauss-distribution
  #note: inputdata is array, centroid is matrix nx1, covariance is matrix nxn
  def gauss_distribution(inputdata, param)
    #calculate gauss mix
    covariance = param[:covariance]
    centroid = param[:centroid]
    input_mtx = Matrix[[inputdata[0]], [inputdata[1]]]
    scalar = 1/(Math.sqrt((2*Math::PI)**2) * Math.sqrt(covariance.determinant)) 
    mtx = substract_by_mean(input_mtx, centroid)
    ret = scalar*exponential_matrix(((-1/2)*mtx.transpose()*covariance.inverse()*mtx))
    return (scalar)*exponential_matrix(((-0.5)*mtx.transpose()*covariance.inverse()*mtx))
  end

  def responsibility(inputdata, cluster_num)
    param = @params[cluster_num]
    singleval = param[:coffmix] * gauss_distribution(inputdata, param)
    sum = 0
    for j in 1..@dimensions do
      _tparam = @param[j]
      sum = sum + _tparam[:coffmix] * gauss_distribution(inputdata, _tparam)
    end
    return singleval / sum
  end

  def loglikelihood
    n = @dataset.length
    k = @dimensions
    ret = 0.0
    for i in 0..(n-1) do
      _temp = 0
      for j in 0..(k-1) do
        gauss = matrix_to_float(gauss_distribution(@dataset[i], @params[j]))
        _temp = _temp + @params[j][:coffmix] * gauss
      end
      ret = ret + Math.log(_temp)
    end
    return ret
  end

  ###############main process
  #initilize, expectation
  #maximization
  #plotting....
  ###########################
  def initialize
    @converged = false
    @dataset = []
    @dimensions = 2 ##fixed num
    @params = []
    @reponsibilities = []

    File.readlines("data10000.txt").each do |line|
      num1, num2 = line.split(",")
      @dataset << [num1.to_f, num2.to_f]
    end
    
    @dataset_length = @dataset.length
    #@mean = @dataset_x.inject(0){|sum, el| sum + el }.to_f / @dataset_x.length
  
    clusters = kmeans(@dataset)
     
    @dimensions.times {|i|
      @params[i] = {}     
      @params[i][:covariance] = covariance(make_matrix(clusters[:clusters][i]))
      @params[i][:centroid] = clusters[:mean][i]
      @params[i][:coffmix] = clusters[:clusters][i].length.to_f / @dataset.length.to_f
      puts @params[i][:covariance]
      puts @params[i][:centroid]
      puts @params[i][:coffmix]
    }
  end
 
  def e_step
     n = @dataset.length
     k = @dimensions
     
     for i in 1..n do
      for j in 1..k do
        (@reponsibilities[i] ||= [])[j] = responsibility(@dataset[i], j)
      end
     end
  end
  
  def m_step
    #TODO: execute m_step
  end 

  def main_loop
    threshold = 1e-6
    dif = 1
    while (dif > threshold) do
      ll_old = loglikelihood
      e_step
      m_step
      ll_new = loglikelihood
      dif = ll_new - ll_old
    end 
  end


  def test
    input = [1.0, 2.0]
    param = {:covariance => Matrix[[-1,-1],[0,-1]], :coffmix => 0.5, :centroid => [2,2]}
    #puts gauss_distribution(input, param)
    mtx = Matrix[[1,2],[1,2]]
  end

  if $0 == __FILE__
    proj = EMAlgorithm.new 
    proj.test
    kmeans_ret = proj.kmeans(proj.dataset)
    proj.main_loop 
    cluster1 = kmeans_ret[:clusters][0]
    cluster2 = kmeans_ret[:clusters][1]
    Gnuplot.open do |gp|
      Gnuplot::Plot.new(gp) do |plot|
        plot.title "em"
        plot.xlabel 'x'
        plot.ylabel 'y'

        y = proj.dataset.collect {|x| x[0]}
        x = proj.dataset.collect {|x| x[1]}

        plot.data << Gnuplot::DataSet.new([x,y]) do |ds|
          ds.with = 'points'
          ds.notitle
        end
      end
    end
  end

end
