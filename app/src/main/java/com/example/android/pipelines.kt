package com.example.android

import android.content.Context
import android.content.res.Resources
import android.graphics.Bitmap
import android.graphics.Rect
import androidx.camera.core.ImageProxy
import org.jetbrains.kotlinx.dl.api.inference.FlatShape
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import org.jetbrains.kotlinx.dl.impl.preprocessing.crop
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModelHub
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.onnx.inference.executionproviders.ExecutionProvider.CPU
import org.jetbrains.kotlinx.dl.onnx.inference.inferUsing
import org.jetbrains.kotlinx.dl.onnx.inference.objectdetection.SSDLikeModel
import org.jetbrains.kotlinx.dl.onnx.inference.objectdetection.detectObjects


interface InferencePipeline {
    fun analyze(image: ImageProxy, confidenceThreshold: Float): Prediction?
    fun close()
}

enum class Tasks(val descriptionId: Int) {
    ObjectDetection(R.string.model_type_object_detection),
}

enum class Pipelines(val task: Tasks, val descriptionId: Int) {
    SSDMobilenetV1(Tasks.ObjectDetection, R.string.pipeline_ssd_mobilenet_v1) {
        override fun createPipeline(hub: ONNXModelHub, resources: Resources): InferencePipeline {
            return DetectionPipeline(ONNXModels.ObjectDetection.SSDMobileNetV1.pretrainedModel(hub))
        }
    };
    abstract fun createPipeline(hub: ONNXModelHub, resources: Resources): InferencePipeline
}

internal class DetectionPipeline(private val model: SSDLikeModel) : InferencePipeline {
    override fun analyze(image: ImageProxy, confidenceThreshold: Float): Prediction? {
        val detections = model.inferUsing(CPU()) {
            it.detectObjects(image, -1)
        }.filter { it.probability >= confidenceThreshold }
        if (detections.isEmpty()) return null

        return PredictedObject(detections)
    }

    override fun close() {
        model.close()
    }

    class PredictedObject(private val detections: List<DetectedObject>) : Prediction {
        override val shapes: List<FlatShape<*>> get() = detections
        override val confidence: Float get() = detections.first().probability
        override fun getText(context: Context): String {
            val singleObject = detections.singleOrNull()
            if (singleObject != null) return singleObject.label ?: ""
            return context.getString(R.string.label_objects, detections.size)
        }
    }
}

private fun <I> Operation<I, Bitmap>.cropRect(rect: Rect): Operation<I, Bitmap> {
    return crop {
        x = rect.left
        y = rect.top
        width = rect.width()
        height = rect.height()
    }
}

